import os
import numpy as np
import torch
import logging

import manotorch
from manotorch.manolayer import ManoLayer
from dev_fn.transform.transform_np import tslrot6d_to_transf_np
from dev_fn.transform.transform import transf_point_array, tslrot6d_to_transf
from dev_fn.transform.rotation_np import rot6d_to_rotmat_np
from dev_fn.transform.rotation import rot6d_to_rotmat, rotmat_to_quat
from pytorch3d.structures import Meshes
from .loss.chamfer_distance import point2point_signed

_logger = logging.getLogger(__name__)


class InteractionSegmentExtraLoss(torch.nn.Module):
    def __init__(self, mano_path, loss_cfg, use_pc=False):
        super().__init__()

        self.mano_layer_rh = ManoLayer(
            mano_assets_root=mano_path,
            rot_mode="quat",
            side="right",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        )
        self.mano_layer_lh = ManoLayer(
            mano_assets_root=mano_path,
            rot_mode="quat",
            side="left",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        )

        self.register_buffer("vpe", torch.from_numpy(np.load(loss_cfg["vpe_path"])).to(torch.long))
        self.register_buffer("v_weights", torch.from_numpy(np.load(loss_cfg["c_weight_path"])).to(torch.float32))
        self.register_buffer("v_weights2", torch.pow(self.v_weights, 1.0 / 2.5))
        self.register_buffer("contact_v", self.v_weights > 0.8)

        self.enable_rec_joint_loss = loss_cfg["coef_rec_joint_loss"] > 0.0
        self.enable_rec_vert_loss = loss_cfg["coef_rec_vert_loss"] > 0.0
        self.enable_edge_len_loss = loss_cfg["coef_edge_len_loss"] > 0.0
        self.use_pc = use_pc
        self.enable_dist_h_loss = loss_cfg["coef_dist_h_loss"] > 0.0
        self.enable_dist_o_loss = loss_cfg["coef_dist_o_loss"] > 0.0

        self.coef_rec_joint_loss = loss_cfg["coef_rec_joint_loss"]
        self.coef_rec_vert_loss = loss_cfg["coef_rec_vert_loss"]
        self.coef_edge_len_loss = loss_cfg["coef_edge_len_loss"]
        self.coef_dist_h_loss = loss_cfg["coef_dist_h_loss"]
        self.coef_dist_o_loss = loss_cfg["coef_dist_o_loss"]

    def retrieve_hand_faces(self, hand_side):
        if hand_side == "rh":
            return self.mano_layer_rh.th_faces
        elif hand_side == "lh":
            return self.mano_layer_lh.th_faces
        else:
            raise ValueError(f"unexpected hand_side: {hand_side}")

    def recover_mano_from_pose_repr(self, pose_repr, shape, hand_side):
        seqlen = shape.shape[0]
        tsl = pose_repr[:, 0:3]  # (seqlen, 3)
        pose_rot6d = pose_repr[:, 3:99]  # (seqlen, 96)
        pose_rot6d = pose_rot6d.reshape((seqlen, 16, 6))  # (seqlen, 16, 6)
        pose_rotmat = rot6d_to_rotmat(pose_rot6d)  # (seqlen, 16, 4, 4)
        pose_quat = rotmat_to_quat(pose_rotmat)  # (seqlen, 16, 4)
        if hand_side == "rh":
            mano_out = self.mano_layer_rh(pose_coeffs=pose_quat, betas=shape)
        elif hand_side == "lh":
            mano_out = self.mano_layer_lh(pose_coeffs=pose_quat, betas=shape)
        else:
            raise ValueError(f"unexpected hand_side: {hand_side}")
        hand_verts = mano_out.verts + tsl.unsqueeze(1)
        hand_joints = mano_out.joints + tsl.unsqueeze(1)
        return hand_verts, hand_joints

    def _edges_for(self, x, vpe):
        return x[:, vpe[:, 0]] - x[:, vpe[:, 1]]

    def forward(self, model_output, batch):
        batch_size = model_output.shape[0]
        seq_len = model_output.shape[3]

        loss_rec_joint = 0.0
        loss_rec_vert = 0.0
        loss_edge_len = 0.0
        loss_dist_h = 0.0
        loss_dist_o = 0.0

        for batch_offset in range(batch_size):
            # decode pose from model and from gt
            hand_side = batch["hand_side"][batch_offset]
            shape = batch["shape"][batch_offset]
            obj_list = batch["obj_list"][batch_offset]
            if self.use_pc:
                obj_verts_list = batch["obj_pointcloud"][batch_offset]
            else:
                obj_verts_list = batch["obj_verts"][batch_offset]
            obj_traj = batch["obj_traj"][batch_offset]
            mask = batch["mask"][batch_offset]  # [seqlen, ]
            with torch.no_grad():
                mask_coef = float(mask.shape[0] / torch.sum(mask))

            pose_repr_gt = batch["pose_repr"][batch_offset]
            pose_repr_pred = model_output[batch_offset].permute((2, 0, 1)).squeeze(-1)  # [seqlen, 99]
            verts_gt, joints_gt = self.recover_mano_from_pose_repr(pose_repr_gt, shape, hand_side)  # [seqlen, n, 3]
            verts_pred, joints_pred = self.recover_mano_from_pose_repr(pose_repr_pred, shape, hand_side)
            mesh_gt = Meshes(verts=verts_gt, faces=self.retrieve_hand_faces(hand_side).unsqueeze(0))
            mesh_pred = Meshes(verts=verts_pred, faces=self.retrieve_hand_faces(hand_side).unsqueeze(0))
            normals_gt = mesh_gt.verts_normals_packed().view((-1, 778, 3))
            normals_pred = mesh_pred.verts_normals_packed().view((-1, 778, 3))

            # joint loss
            if self.enable_rec_joint_loss:
                joint_dist_sq = torch.sum(torch.pow(joints_pred - joints_gt, exponent=2), dim=-1)  # [seqlen, n]
                joint_dist_sq = joint_dist_sq * mask.unsqueeze(1)
                joint_loss = mask_coef * torch.mean(joint_dist_sq)
                loss_rec_joint = loss_rec_joint + joint_loss

            # verts loss
            if self.enable_rec_vert_loss:
                vert_dist_sq = torch.sum(torch.pow(verts_pred - verts_gt, exponent=2), dim=-1)  # [seqlen, n]
                vert_dist_sq = vert_dist_sq * mask.unsqueeze(1)
                vert_loss = mask_coef * torch.mean(
                    torch.einsum(
                        "ij,j->ij",
                        vert_dist_sq,
                        torch.pow(self.v_weights, 2),
                    )
                )  # [1]
                loss_rec_vert = loss_rec_vert + vert_loss

            # edge len loss
            if self.enable_edge_len_loss:
                edge_len_pred = self._edges_for(verts_pred, self.vpe)  # [seqlen, ne, 3]
                edge_len_gt = self._edges_for(verts_gt, self.vpe)  # [seqlen, ne, 3]
                edge_diff = torch.abs(edge_len_pred - edge_len_gt)
                edge_loss = mask_coef * torch.mean(edge_diff * mask.unsqueeze(1).unsqueeze(2))
                loss_edge_len = loss_edge_len + edge_loss

            if self.enable_dist_h_loss or self.enable_dist_o_loss:
                num_obj = len(obj_list)
                for obj_offset, obj_id in enumerate(obj_list):
                    obj_verts_can_sel = torch.from_numpy(obj_verts_list[obj_offset]).to(verts_gt)
                    obj_traj_sel = obj_traj[obj_offset]
                    obj_traj_sel = tslrot6d_to_transf(obj_traj_sel)
                    obj_verts_sel = transf_point_array(
                        obj_traj_sel, obj_verts_can_sel.unsqueeze(0).expand((seq_len, -1, -1))
                    )
                    n_obj_verts = obj_verts_sel.shape[1]

                    o2h_signed, h2o, _ = point2point_signed(verts_pred, obj_verts_sel, normals_pred)
                    o2h_signed_gt, h2o_gt, o2h_idx = point2point_signed(verts_gt, obj_verts_sel, normals_gt)
                    w_dist = (o2h_signed_gt < 0.01) * (o2h_signed_gt > -0.005)
                    w_dist_neg = o2h_signed < 0.0
                    w = torch.ones([seq_len, n_obj_verts]).to(verts_gt.device)
                    w[~w_dist] = 0.1  # less weight for far away vertices
                    w[w_dist_neg] = 1.5  # more weight for penetration

                    # dist_h
                    if self.enable_dist_h_loss:
                        dist_h = mask_coef * torch.mean(
                            torch.einsum("ij,j->ij", torch.abs(h2o.abs() - h2o_gt.abs()), self.v_weights2)
                            * mask.unsqueeze(1)
                        )
                        loss_dist_h = loss_dist_h + (1.0 / num_obj) * dist_h

                    # dist_o
                    if self.enable_dist_o_loss:
                        dist_o = mask_coef * torch.mean(
                            torch.einsum("ij,ij->ij", torch.abs(o2h_signed - o2h_signed_gt), w) * mask.unsqueeze(1)
                        )
                        loss_dist_o = loss_dist_o + (1.0 / num_obj) * dist_o

            # TODO: smoothness loss
            ## penalize the diviergence of tsl and rot diff

        loss = 0.0
        loss = loss + self.coef_rec_joint_loss * loss_rec_joint
        loss = loss + self.coef_rec_vert_loss * loss_rec_vert
        loss = loss + self.coef_edge_len_loss * loss_edge_len
        loss = loss + self.coef_dist_h_loss * loss_dist_h
        loss = loss + self.coef_dist_o_loss * loss_dist_o
        loss_dict = {
            "loss": loss,
            "rec_joint": loss_rec_joint,
            "rec_vert": loss_rec_vert,
            "edge_len": loss_edge_len,
            "dist_h": loss_dist_h,
            "dist_o": loss_dist_o,
        }
        return loss, loss_dict
