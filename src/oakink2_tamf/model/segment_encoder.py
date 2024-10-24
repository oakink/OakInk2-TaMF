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


class SegmentEncoder(torch.nn.Module):
    def __init__(
        self,
        output_dim,
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
    ):
        super().__init__()

        self.output_dim = output_dim
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

        self.hand_side_process = HandsideProcess(self.latent_dim)
        self.hand_shape_process = HandShapeProcess(self.hand_shape_feats, self.latent_dim)
        self.obj_embed_process = ObjectEmbedProcess(self.obj_embed_feats, self.latent_dim)
        self.register_buffer("classification_token", torch.zeros(1, 1, self.latent_dim))

        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.obj_input_process = ObjectInputProcess(self.obj_input_feats, self.latent_dim)
        self.input_merge = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
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

    def forward(self, batch):
        x_in = batch["pose_repr"]
        batch_size = x_in.shape[0]

        # handle embed
        emb_list = []

        emb_handside = self.hand_side_process(batch["hand_side"])  # [1, bs, d]
        emb_list.append(emb_handside)
        emb_shape = self.hand_shape_process(batch["shape"])  # [1, bs, d]
        emb_list.append(emb_shape)

        emb_obj = self.obj_embed_process(batch["obj_embedding"])  # [1, bs, d]
        emb_list.append(emb_obj)

        emb = torch.cat(emb_list, dim=0)  # [3, bs, d]
        emb = torch.nan_to_num(emb)
        emb_prefix_len = emb.shape[0]

        # handle object
        hand_traj = self.input_process(x_in)  # [seq_len, bs, d]
        object_input = self.obj_input_process(batch["obj_traj"])  # [seq_len, bs, d]
        
        # brute force merger
        merged_input = torch.cat((hand_traj, object_input), dim=-1)
        x = self.input_merge(merged_input)  # [seq_len, bs, d]
        x = torch.nan_to_num(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x, self.classification_token.expand(-1, batch_size, -1)), axis=0)  # [3+seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [3+seqlen+1, bs, d]
        encoding = self.seqTransEncoder(xseq)[-1:]  # [1, bs, d]
        activation = self.output_process(encoding)  # [bs, nframes, input_di,]

        res = {"encoding": encoding, "activation": activation}

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
    def __init__(self, output_feats, latent_dim):
        super().__init__()
        self.output_feats = output_feats
        self.latent_dim = latent_dim
        # use an mlp
        self.poseFinal = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.output_feats),
        )

    def forward(self, output):
        output = output.squeeze(0)
        bs, d = output.shape
        output = self.poseFinal(output)  # [bs, output_dim]
        return output
