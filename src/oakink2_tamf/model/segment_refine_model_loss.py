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


class SegmentRefineModelLoss(torch.nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()

        self.register_buffer("vpe", torch.from_numpy(np.load(loss_cfg["vpe_path"])).to(torch.long))
        self.register_buffer("v_weights", torch.from_numpy(np.load(loss_cfg["c_weight_path"])).to(torch.float32))
        self.register_buffer("v_weights2", torch.pow(self.v_weights, 1.0 / 2.5))
        self.register_buffer("contact_v", self.v_weights > 0.8)

        self.enable_rec_joint_loss = loss_cfg["coef_rec_joint_loss"] > 0.0
        self.enable_rec_vert_loss = loss_cfg["coef_rec_vert_loss"] > 0.0
        self.enable_dist_h_loss = loss_cfg["coef_dist_h_loss"] > 0.0

        self.coef_rec_joint_loss = loss_cfg["coef_rec_joint_loss"]
        self.coef_rec_vert_loss = loss_cfg["coef_rec_vert_loss"]
        self.coef_dist_h_loss = loss_cfg["coef_dist_h_loss"]

    def forward(self, output, batch):
        loss_rec_joint = 0.0
        loss_rec_vert = 0.0
        loss_dist_h = 0.0

        mask = batch["mask"]  # [bs, seqlen, ]
        with torch.no_grad():
            mask_coef = float(mask.shape[1]) / torch.sum(mask, dim=1)  # [bs, ]

        # joint loss
        if self.enable_rec_joint_loss:
            refine_hand_joints = output["refine_hand_joints"]
            target_hand_joints = output["target_hand_joints"]
            joint_dist_sq = torch.sum(torch.pow(refine_hand_joints - target_hand_joints, 2), dim=-1)  # [bs, seqlen, nj]
            joint_dist_sq = joint_dist_sq * mask.unsqueeze(-1)  # [bs, seqlen, nj]
            joint_loss = mask_coef * torch.mean(joint_dist_sq, dim=(-2, -1))  # [bs, ]
            joint_loss = torch.mean(joint_loss)
            loss_rec_joint = loss_rec_joint + joint_loss

        if self.enable_rec_vert_loss:
            refine_hand_verts = output["refine_hand_verts"]
            target_hand_verts = output["target_hand_verts"]
            vert_dist_sq = torch.sum(torch.pow(refine_hand_verts - target_hand_verts, 2), dim=-1)  # [bs, seqlen, nv]
            vert_dist_sq = vert_dist_sq * mask.unsqueeze(-1)  # [bs, seqlen, nv]
            vert_dist_sq = torch.einsum("bij,j->bij", vert_dist_sq, torch.pow(self.v_weights, 2))  # [bs, seqlen, nv]
            vert_loss = mask_coef * torch.mean(vert_dist_sq, dim=(-2, -1))  # [bs, ]
            vert_loss = torch.mean(vert_loss)
            loss_rec_vert = loss_rec_vert + vert_loss

        if self.enable_dist_h_loss:
            refine_h2o_dist = output["refine_h2o_dist"]  # [bs, seqlen, nv]
            target_h2o_dist = output["target_h2o_dist"]  # [bs, seqlen, nv]
            dist_h = torch.abs(torch.abs(refine_h2o_dist) - torch.abs(target_h2o_dist))  # [bs, seqlen, nv]
            dist_h = dist_h * mask.unsqueeze(-1)  # [bs, seqlen, nv]
            dist_h = torch.einsum("bij,j->bij", dist_h, self.v_weights2) # [bs, seqlen, nv]
            dist_h_loss = mask_coef * torch.mean(dist_h, dim=(-2, -1))
            dist_h_loss = torch.mean(dist_h_loss)
            loss_dist_h = loss_dist_h + dist_h_loss

        loss = 0.0
        loss = loss + self.coef_rec_joint_loss * loss_rec_joint
        loss = loss + self.coef_rec_vert_loss * loss_rec_vert
        loss = loss + self.coef_dist_h_loss * loss_dist_h
        loss_dict = {
            "loss": loss,
            "rec_joint": loss_rec_joint,
            "rec_vert": loss_rec_vert,
            "dist_h": loss_dist_h,
        }
        return loss, loss_dict