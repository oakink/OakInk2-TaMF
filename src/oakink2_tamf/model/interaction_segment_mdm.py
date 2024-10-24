import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


_logger = logging.getLogger(__name__)


class InterationSegmentMDM(nn.Module):
    def __init__(
        self,
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
        clip_dim=512,
        clip_version="ViT-B/32",
        **kargs,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim

        self.input_feats = input_dim
        self.obj_input_feats = obj_input_dim

        self.hand_shape_feats = hand_shape_dim
        self.obj_embed_feats = obj_embed_dim

        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.hand_side_process = HandsideProcess(self.latent_dim)
        self.hand_shape_process = HandShapeProcess(self.hand_shape_feats, self.latent_dim)
        self.obj_embed_process = ObjectEmbedProcess(self.obj_embed_feats, self.latent_dim)

        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.obj_input_process = ObjectInputProcess(self.obj_input_feats, self.latent_dim)
        self.input_merge = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        _logger.info("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        _logger.info("EMBED TEXT")
        _logger.info("Loading CLIP...")
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)

        self.output_process = OutputProcess(self.input_feats, self.latent_dim)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device="cpu", jit=False
        )  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(
                bs, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length], dtype=texts.dtype, device=texts.device
            )
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, batch):
        """
        x: [TODO]
        timesteps: [batch_size] (int) for diffusion
        """
        batch_size = x.shape[0]

        emb_list = []
        emb_timestep = self.embed_timestep(timesteps)  # [1, bs, d]
        emb_list.append(emb_timestep)

        enc_text = self.encode_text(batch["text"])
        emb_text = self.embed_text(self.mask_cond(enc_text, force_mask=False))  # [bs, d]
        emb_text = emb_text.reshape((1, batch_size, self.latent_dim))
        emb_list.append(emb_text)

        emb_handside = self.hand_side_process(batch["hand_side"])  # [1, bs, d]
        emb_list.append(emb_handside)
        emb_shape = self.hand_shape_process(batch["shape"])  # [1, bs, d]
        emb_list.append(emb_shape)

        emb_obj = self.obj_embed_process(batch["obj_embedding"])  # [1, bs, d]
        emb_list.append(emb_obj)
        emb = torch.cat(emb_list, dim=0)  # [5, bs, d]
        emb = torch.nan_to_num(emb)
        emb_prefix_len = emb.shape[0]

        hand_traj = self.input_process(x)  # [seq_len, bs, d]
        object_input = self.obj_input_process(batch["obj_traj"])  # [seq_len, bs, d]
        # brute force merger
        merged_input = torch.cat((hand_traj, object_input), dim=-1)
        x = self.input_merge(merged_input)  # [seq_len, bs, d]
        x = torch.nan_to_num(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+5, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+5, bs, d]
        output = self.seqTransEncoder(xseq)[emb_prefix_len:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.output_process(output)  # [bs, input_dim, 1, nframes]
        output = torch.nan_to_num(output)
        return output

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.clip_model.eval()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nfeats, _, nframes = x.shape  # [bs, in_dim, 1, seqlen]
        x = x.permute((3, 0, 1, 2))  # [seqlen, bs, i, 1]
        x = x.reshape((nframes, bs, nfeats))  # [seqlen, bs, i]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class ObjectInputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        # self.hidden_dim = hidden_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)  # TODO: use a MLP

    def forward(self, x):
        bs, nobj, seqlen, nfeat = x.shape
        x = x.permute((0, 2, 1, 3))  # [bs, seqlen, nobj, inp]
        x = self.poseEmbedding(x)  # [bs, seqlen, nobj, d]
        # avg pool
        x = torch.mean(x, dim=2)  # [bs, seqlen, d]
        x = x.permute((1, 0, 2))  # [seqlen, bs, d]
        return x


class ObjectEmbedProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.embedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nobj, dim_obj_feat = x.shape
        x = torch.mean(x, dim=1)  # average -> [bs, dim_obj_feat]
        x = self.embedding(x)  # [bs, d]
        x = x.unsqueeze(0)  # [1, bs, d]
        return x


class HandsideProcess(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        _rh = torch.zeros((self.latent_dim,), dtype=torch.float32)
        self.register_buffer("rh_embed", _rh)
        _lh = torch.zeros((self.latent_dim,), dtype=torch.float32)
        _lh[0] = 1.0
        self.register_buffer("lh_embed", _lh)

    def forward(self, hand_side):
        res = []
        for hs in hand_side:
            if hs == "rh":
                embed = self.rh_embed
            elif hs == "lh":
                embed = self.lh_embed
            else:
                raise ValueError(f"unexpected hand_side: {hs}")
            res.append(embed)
        res = torch.stack(res, dim=0)  # [bs, d]
        res = res.unsqueeze(0)  # [1, bs, d]
        return res


class HandShapeProcess(nn.Module):
    def __init__(self, shape_dim, latent_dim) -> None:
        super().__init__()
        self.shape_dim = shape_dim
        self.latent_dim = latent_dim
        self.shape_embed = nn.Linear(self.shape_dim, self.latent_dim)

    def forward(self, shape):
        # shape [B, SEQLEN, 10]
        shape_avg = torch.mean(shape, dim=1)  # [bs, 10]
        res = self.shape_embed(shape_avg)  # [bs, d]
        res = res.unsqueeze(0)  # [1, bs, d]
        return res


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, input_dim]
        output = output.reshape(nframes, bs, self.input_feats, 1)
        output = output.permute(1, 2, 3, 0)  # [bs, inp, 1, nframes]
        return output
