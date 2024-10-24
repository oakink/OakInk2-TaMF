import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interaction_segment_mdm import (
    PositionalEncoding,
    HandsideProcess,
    HandShapeProcess,
    ObjectEmbedProcess,
    ObjectInputProcess,
)
from manotorch.manolayer import ManoLayer
from dev_fn.transform.rotation import rot6d_to_rotmat, rotmat_to_quat
from dev_fn.transform.transform import tslrot6d_to_transf, transf_point_array
from pytorch3d.structures import Meshes
from .loss.chamfer_distance import point2point_signed


class SegmentRefineModel(torch.nn.Module):
    def __init__(
        self,
        mano_path,
        input_dim=99,
        obj_input_dim=9,
        hand_shape_dim=10,
        obj_embed_dim=768,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        use_pc=False,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation

        self.input_feats = input_dim
        self.obj_input_feats = obj_input_dim

        self.hand_shape_feats = hand_shape_dim
        self.obj_embed_feats = obj_embed_dim

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

        self.hand_side_process = HandsideProcess(self.latent_dim)
        self.hand_shape_process = HandShapeProcess(self.hand_shape_feats, self.latent_dim)
        self.obj_embed_process = ObjectEmbedProcess(self.obj_embed_feats, self.latent_dim)

        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.obj_input_process = ObjectInputProcess(self.obj_input_feats, self.latent_dim)
        self.use_pc = use_pc
        self.h2o_dist_input_process = H2ODistInputProcess(778, self.latent_dim)

        self.input_merge = nn.Sequential(
            nn.Linear(self.latent_dim * 3, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        self.output_process = OutputProcess(self.input_feats, self.latent_dim)

    def retrieve_hand_faces(self, hand_side):
        if hand_side == "rh":
            return self.mano_layer_rh.th_faces
        elif hand_side == "lh":
            return self.mano_layer_lh.th_faces
        else:
            raise ValueError(f"unexpected hand_side: {hand_side}")

    def batch_recover_mano_from_pose_repr(self, batch_pose_repr, batch_shape, batch_hand_side):
        # batch_pose_repr (bs, seqlen, 99)
        # batch_shape (bs, seqlen, 10)
        # batch_hand_side (bs, )
        batch_size = batch_shape.shape[0]
        hand_verts, hand_joints, hand_normals = [], [], []
        for batch_offset in range(batch_size):
            pose_repr = batch_pose_repr[batch_offset]
            shape = batch_shape[batch_offset]
            hand_side = batch_hand_side[batch_offset]

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
            _v = mano_out.verts + tsl.unsqueeze(1)
            _j = mano_out.joints + tsl.unsqueeze(1)
            _mesh = Meshes(verts=_v, faces=self.retrieve_hand_faces(hand_side).unsqueeze(0))
            _normal = _mesh.verts_normals_packed().view((-1, 778, 3))
            hand_verts.append(_v)
            hand_joints.append(_j)
            hand_normals.append(_normal)
        hand_verts = torch.stack(hand_verts, dim=0)
        hand_joints = torch.stack(hand_joints, dim=0)
        hand_normals = torch.stack(hand_normals, dim=0)
        return hand_verts, hand_joints, hand_normals

    def multi_object_h2o_dist(
        self, batch_hand_verts, batch_hand_normals, batch_obj_list, batch_obj_traj, batch_obj_verts_list
    ):
        batch_size = batch_hand_verts.shape[0]
        seq_len = batch_hand_verts.shape[1]
        _h2o_list = []
        for batch_offset in range(batch_size):
            hand_verts = batch_hand_verts[batch_offset]  # (seqlen, nhv, 3)
            hand_normals = batch_hand_normals[batch_offset]  # (seqlen, nhv, 3)

            obj_traj = batch_obj_traj[batch_offset]  # (nobj, seqlen, 9)
            obj_num = len(batch_obj_list[batch_offset])
            obj_verts_arr = torch.from_numpy(batch_obj_verts_list[batch_offset]).to(hand_verts)  # (nobj, 8192, 3)
            obj_traj_transf = tslrot6d_to_transf(obj_traj)  # (nobj, seqlen, 4, 4)
            _obj_list = []
            for obj_offset in range(obj_num):
                obj_verts_sel = obj_verts_arr[obj_offset]
                obj_traj_transf_sel = obj_traj_transf[obj_offset]
                obj_verts_transf_sel = transf_point_array(
                    obj_traj_transf_sel, obj_verts_sel.unsqueeze(0).expand((seq_len, -1, -1))
                )  # [seqlen, nv, 3]
                _obj_list.append(obj_verts_transf_sel)
            obj_verts_transf = torch.cat(_obj_list, dim=1)  # (seqlen, _, 3)
            _, _h2o, _ = point2point_signed(hand_verts, obj_verts_transf, hand_normals)
            _h2o_list.append(_h2o)
        h2o = torch.stack(_h2o_list, dim=0)  # (bs, seqlen, nhv)
        return h2o

    def forward(self, batch):
        x_in = batch["sample_pose_repr"]
        batch_size = x_in.shape[0]

        # handle embed
        emb_list = []

        emb_handside = self.hand_side_process(batch["hand_side"])  # [1, bs, d]
        emb_list.append(emb_handside)
        emb_shape = self.hand_shape_process(batch["shape"])  # [1, bs, d]
        emb_list.append(emb_shape)

        emb_obj = self.obj_embed_process(batch["obj_embedding"])  # [1, bs, d]
        emb_list.append(emb_obj)
        emb = torch.cat(emb_list, dim=0)  # [5, bs, d]
        emb = torch.nan_to_num(emb)
        emb_prefix_len = emb.shape[0]

        # handle object
        hand_traj = self.input_process(x_in)  # [seq_len, bs, d]
        object_input = self.obj_input_process(batch["obj_traj"])  # [seq_len, bs, d]

        # handle local hand object distance
        hand_verts, hand_joints, hand_normals = self.batch_recover_mano_from_pose_repr(
            x_in, batch["shape"], batch["hand_side"]
        )
        if self.use_pc:
            obj_verts_list = batch["obj_pointcloud"]
        else:
            obj_verts_list = batch["obj_verts"]

        h2o_dist = self.multi_object_h2o_dist(
            hand_verts, hand_normals, batch["obj_list"], batch["obj_traj"], obj_verts_list
        )
        h2o_dist_input = self.h2o_dist_input_process(h2o_dist)

        # brute force merger
        merged_input = torch.cat((hand_traj, object_input, h2o_dist_input), dim=-1)
        x = self.input_merge(merged_input)  # [seq_len, bs, d]
        x = torch.nan_to_num(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+5, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+5, bs, d]
        output = self.seqTransEncoder(xseq)[emb_prefix_len:]  # [seqlen, bs, d]
        output = self.output_process(output)  # [bs, nframes, input_di,]
        output = x_in + output  # use res value
        output = torch.nan_to_num(output)

        # compute gt for loss comp
        refine_hand_verts, refine_hand_joints, refine_hand_normals = self.batch_recover_mano_from_pose_repr(
            output, batch["shape"], batch["hand_side"]
        )
        h2o_dist_refine = self.multi_object_h2o_dist(
            refine_hand_verts, refine_hand_normals, batch["obj_list"], batch["obj_traj"], obj_verts_list
        )
        with torch.no_grad():
            target_hand_verts, target_hand_joints, target_hand_normals = self.batch_recover_mano_from_pose_repr(
                batch["pose_repr"], batch["shape"], batch["hand_side"]
            )
            h2o_dist_gt = self.multi_object_h2o_dist(
                target_hand_verts, target_hand_normals, batch["obj_list"], batch["obj_traj"], obj_verts_list
            )

        res = {
            "refine_pose_repr": output,
            "refine_hand_verts": refine_hand_verts,
            "refine_hand_joints": refine_hand_joints,
            "refine_hand_normals": refine_hand_normals,
            "refine_h2o_dist": h2o_dist_refine,
            "target_hand_verts": target_hand_verts,
            "target_hand_joints": target_hand_joints,
            "target_hand_normals": target_hand_normals,
            "target_h2o_dist": h2o_dist_gt,
            "sample_hand_verts": hand_verts,
            "sample_hand_joints": hand_joints,
            "sample_hand_normals": hand_normals,
            "sample_h2o_dist": h2o_dist,
        }

        return res


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nframes, nfeats = x.shape  # [bs, seq_len, in_dim]
        x = x.permute((1, 0, 2))  # [seqlen, bs, i]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class H2ODistInputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()

        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nframes, nfeats = x.shape  # [bs, seq_len, in_dim]
        x = x.permute((1, 0, 2))  # [seqlen, bs, i]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, input_dim]
        output = output.permute(1, 0, 2)  # [bs, seqlen, input_dim]
        return output
