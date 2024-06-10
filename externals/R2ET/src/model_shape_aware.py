import os

import torch
import torch.nn as nn
import numpy as np
import time
import math

from src.forward_kinematics import FK
from src.linear_blend_skin import linear_blend_skinning
from src.ops import qlinear, q_mul_q
from torch import atan2
from torch import asin
from collections import OrderedDict
import trimesh

import scipy


class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, out_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, n, 3, h, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0, :, :, :, :], qkv[1, :, :, :, :], qkv[2, :, :, :, :]

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)

        out = self.nn1(out)
        out = self.do1(out)

        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hid_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hid_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        if dim == mlp_dim:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Residual(Attention(dim, mlp_dim, heads=heads, dropout=dropout)),
                            Residual(
                                LayerNormalize(
                                    mlp_dim,
                                    MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout),
                                )
                            ),
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Attention(dim, mlp_dim, heads=heads, dropout=dropout),
                            Residual(
                                LayerNormalize(
                                    mlp_dim,
                                    MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout),
                                )
                            ),
                        ]
                    )
                )

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=22):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with " "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, n, self.dim)``
        """

        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:, 0 : emb.size(1), :]
        emb = self.dropout(emb)
        return emb


class QuatEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, hidden_channels, kp):
        super(QuatEncoder, self).__init__()

        self.num_joint = num_joint
        self.token_linear = nn.Linear(4, token_channels)
        self.trans1 = Transformer(token_channels, 1, 4, hidden_channels, 1 - kp)

    def forward(self, pose_t):
        # pose_t: bs joint, 4
        token_q = self.token_linear(pose_t)
        embed_q = self.trans1(token_q)

        return embed_q


class SeklEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, kp):
        super(SeklEncoder, self).__init__()

        self.num_joint = num_joint
        self.token_linear = nn.Linear(3, token_channels)
        self.trans1 = Transformer(token_channels, 1, 2, embed_channels, 1 - kp)

    def forward(self, skel):
        # bs = skel.shape[0]
        token_s = self.token_linear(skel)
        embed_s = self.trans1(token_s)

        return embed_s


class ShapeEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, kp):
        super(ShapeEncoder, self).__init__()

        self.num_joint = num_joint
        self.token_linear = nn.Linear(3, token_channels)
        self.trans1 = Transformer(token_channels, 1, 2, embed_channels, 1 - kp)

    def forward(self, shape):
        token_s = self.token_linear(shape)
        embed_s = self.trans1(token_s)

        return embed_s


class DeltaDecoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, hidden_channels, kp):
        super(DeltaDecoder, self).__init__()

        self.num_joint = num_joint
        self.q_encoder = QuatEncoder(num_joint, token_channels, hidden_channels, kp)
        self.skel_encoder = SeklEncoder(num_joint, token_channels, embed_channels, kp)
        self.pos_encoder = PositionalEncoding(1 - kp, hidden_channels + (2 * embed_channels))

        self.embed_linear = nn.Linear(hidden_channels + (2 * embed_channels), embed_channels)
        self.embed_acti = nn.ReLU()
        self.embed_drop = nn.Dropout(1 - kp)
        self.delta_linear = nn.Linear(embed_channels, 4)

    def forward(self, q_t, skelA, skelB):
        q_embed = self.q_encoder(q_t)
        skelA_embed = self.skel_encoder(skelA)
        skelB_embed = self.skel_encoder(skelB)

        cat_embed = torch.cat([q_embed, skelA_embed, skelB_embed], dim=-1)  # bs n c
        pos_embed = self.pos_encoder(cat_embed)

        embed = self.embed_drop(self.embed_acti(self.embed_linear(pos_embed)))
        deltaq_t = self.delta_linear(embed)

        return deltaq_t


class DeltaShapeDecoder(nn.Module):
    def __init__(self, num_joint, hidden_channels, kp):
        super(DeltaShapeDecoder, self).__init__()

        self.num_joint = num_joint

        self.joint_linear1 = nn.Linear(10 * 22, hidden_channels)
        self.joint_acti1 = nn.ReLU()
        self.joint_drop1 = nn.Dropout(p=1 - kp)
        self.joint_linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.joint_acti2 = nn.ReLU()
        self.joint_drop2 = nn.Dropout(p=1 - kp)

        self.delta_linear = qlinear(hidden_channels, 4 * num_joint)

    def forward(self, shapeA, shapeB, x):
        bs = shapeB.shape[0]
        x_cat = torch.cat([shapeA, shapeB, x], dim=-1)
        x_cat = x_cat.view((bs, -1))

        x_embed = self.joint_drop1(self.joint_acti1(self.joint_linear1(x_cat)))
        x_embed = self.joint_drop2(self.joint_acti2(self.joint_linear2(x_embed)))
        deltaq_t = self.delta_linear(x_embed)

        return deltaq_t


class WeightsDecoder(nn.Module):
    def __init__(self, num_joint, hidden_channels, kp):
        super(WeightsDecoder, self).__init__()

        self.num_joint = num_joint

        self.joint_linear1 = nn.Linear(10 * num_joint, 2 * hidden_channels)
        self.joint_acti1 = nn.ReLU()
        self.joint_drop1 = nn.Dropout(p=1 - kp)
        self.joint_linear2 = nn.Linear(2 * hidden_channels, 2 * hidden_channels)
        self.joint_acti2 = nn.ReLU()
        self.joint_drop2 = nn.Dropout(p=1 - kp)
        self.joint_linear3 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.joint_acti3 = nn.ReLU()
        self.joint_drop3 = nn.Dropout(p=1 - kp)

        self.weights_linear = nn.Linear(hidden_channels, num_joint)
        self.weights_acti = nn.Sigmoid()

    def forward(self, refB, shapeB, x):
        bs = refB.shape[0]
        x_cat = torch.cat([refB, shapeB, x], dim=-1)
        x_cat = x_cat.view((bs, -1))

        x_embed = self.joint_drop1(self.joint_acti1(self.joint_linear1(x_cat)))
        x_embed = self.joint_drop2(self.joint_acti2(self.joint_linear2(x_embed)))
        x_embed = self.joint_drop3(self.joint_acti3(self.joint_linear3(x_embed)))

        weights = self.weights_acti(self.weights_linear(x_embed))

        return weights


class MotionDis(nn.Module):
    def __init__(self, kp):
        super(MotionDis, self).__init__()
        pad = int((4 - 1) / 2)

        self.seq = nn.Sequential(
            OrderedDict(
                [
                    ("dropout", nn.Dropout(p=1 - kp)),
                    ("h0", nn.Conv1d(3, 16, kernel_size=4, padding=pad, stride=2)),
                    ("acti0", nn.LeakyReLU(0.2)),
                    ("h1", nn.Conv1d(16, 32, kernel_size=4, padding=pad, stride=2)),
                    ("bn1", nn.BatchNorm1d(32)),
                    ("acti1", nn.LeakyReLU(0.2)),
                    ("h2", nn.Conv1d(32, 64, kernel_size=4, padding=pad, stride=2)),
                    ("bn2", nn.BatchNorm1d(64)),
                    ("acti2", nn.LeakyReLU(0.2)),
                    ("h3", nn.Conv1d(64, 64, kernel_size=4, padding=pad, stride=2)),
                    ("bn3", nn.BatchNorm1d(64)),
                    ("acti3", nn.LeakyReLU(0.2)),
                    ("h4", nn.Conv1d(64, 1, kernel_size=3, stride=2)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        # x: bs 3 T
        bs = x.size(0)
        y = self.seq(x)
        return y.view(bs, 1)


def normalized(angles):
    lengths = torch.sqrt(torch.sum(torch.square(angles), dim=-1))
    normalized_angle = angles / lengths[..., None]
    return normalized_angle


class RetNet(nn.Module):
    def __init__(
        self,
        num_joint=22,
        token_channels=64,
        hidden_channels_p=256,
        embed_channels_p=128,
        kp=0.8,
    ):
        super(RetNet, self).__init__()
        self.num_joint = num_joint
        self.delta_dec = DeltaDecoder(num_joint, token_channels, embed_channels_p, hidden_channels_p, kp)

        self.delta_leftArm_dec = DeltaShapeDecoder(3, hidden_channels_p, kp)
        self.delta_rightArm_dec = DeltaShapeDecoder(3, hidden_channels_p, kp)
        self.delta_leftLeg_dec = DeltaShapeDecoder(2, hidden_channels_p, kp)
        self.delta_rightLeg_dec = DeltaShapeDecoder(2, hidden_channels_p, kp)

        self.weights_dec = WeightsDecoder(num_joint, hidden_channels_p, kp)

    def forward(
        self,
        seqA,
        seqB,
        skelA,
        skelB,
        shapeA,
        shapeB,
        quatA,
        inp_height,
        tgt_height,
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        parents,
        k=-1,
    ):
        """
        seqA, seqB: bs T joints*3+4
        skelA, skelB: bs T joints*3
        shapeA, shapeB: bs 6
        height: bs 1
        """
        self.parents = parents
        bs, T = seqA.size(0), seqA.size(1)

        parents = torch.from_numpy(parents).cuda(seqA.device)

        t_poseB = torch.reshape(skelB[:, 0, :], [bs, self.num_joint, 3])
        t_poseB = t_poseB * local_std + local_mean
        refB = t_poseB
        refB_feed = skelB[:, 0, :]
        refA_feed = skelA[:, 0, :]
        shapeB = shapeB.view((bs, self.num_joint, 3))
        shapeA = shapeA.view((bs, self.num_joint, 3))

        quatA_denorm = quatA * quat_std[None, :] + quat_mean[None, :]

        delta_qs = []
        B_locals_rt = []
        B_quats_rt = []
        B_quats_base = []
        B_locals_base = []
        B_gates = []
        delta_qg = []

        # manully adjust k for balacing gate
        if k == -1:
            k = 1.0
        leftArm_joints = [14, 15, 16]
        rightArm_joints = [18, 19, 20]
        leftLeg_joints = [6, 7]
        rightLeg_joints = [10, 11]

        """ mapping from A to B frame by frame"""
        for t in range(T):
            qoutA_t = quatA[:, t, :, :]  # motion copy
            qoutA_t_denorm = quatA_denorm[:, t, :, :]

            # delta qs
            refA_feed = refA_feed.view((bs, self.num_joint, 3))
            refB_feed = refB_feed.view((bs, self.num_joint, 3))
            delta1 = self.delta_dec(qoutA_t, refA_feed, refB_feed)
            delta1 = delta1 * quat_std + quat_mean
            delta1 = normalized(delta1)
            delta_qs.append(delta1)

            qB_base = q_mul_q(qoutA_t_denorm, delta1)
            qB_base = qB_base.detach()
            qB_base_norm = (qB_base - quat_mean) / quat_std

            # delta qg
            delta2_leftArm = self.delta_leftArm_dec(shapeA, shapeB, qB_base_norm)
            delta2_leftArm = torch.reshape(delta2_leftArm, [bs, 3, 4])
            delta2_leftArm = delta2_leftArm * quat_std[:, leftArm_joints, :] + quat_mean[:, leftArm_joints, :]
            delta2_leftArm = normalized(delta2_leftArm)

            delta2_rightArm = self.delta_rightArm_dec(shapeA, shapeB, qB_base_norm)
            delta2_rightArm = torch.reshape(delta2_rightArm, [bs, 3, 4])
            delta2_rightArm = delta2_rightArm * quat_std[:, rightArm_joints, :] + quat_mean[:, rightArm_joints, :]
            delta2_rightArm = normalized(delta2_rightArm)

            delta2_leftLeg = self.delta_leftLeg_dec(shapeA, shapeB, qB_base_norm)
            delta2_leftLeg = torch.reshape(delta2_leftLeg, [bs, 2, 4])
            delta2_leftLeg = delta2_leftLeg * quat_std[:, leftLeg_joints, :] + quat_mean[:, leftLeg_joints, :]
            delta2_leftLeg = normalized(delta2_leftLeg)

            delta2_rightLeg = self.delta_rightLeg_dec(shapeA, shapeB, qB_base_norm)
            delta2_rightLeg = torch.reshape(delta2_rightLeg, [bs, 2, 4])
            delta2_rightLeg = delta2_rightLeg * quat_std[:, rightLeg_joints, :] + quat_mean[:, rightLeg_joints, :]
            delta2_rightLeg = normalized(delta2_rightLeg)

            # mask
            delta2 = torch.tensor([1, 0, 0, 0], dtype=torch.float32).cuda(seqA.device).repeat(bs, self.num_joint, 1)
            delta2[:, leftArm_joints, :] = delta2_leftArm
            delta2[:, rightArm_joints, :] = delta2_rightArm
            delta2[:, leftLeg_joints, :] = delta2_leftLeg
            delta2[:, rightLeg_joints, :] = delta2_rightLeg
            delta_qg.append(delta2)

            # balacing gate
            bala_gate = self.weights_dec(refB_feed, shapeB, qB_base_norm)

            qB_hat = q_mul_q(qB_base, delta2)

            one_w = np.random.binomial(1, p=0.4)
            if one_w:
                bala_gate = torch.ones(bala_gate.shape, dtype=torch.float32).cuda(seqA.device)

            qB_t = torch.lerp(qB_base, qB_hat, bala_gate[:, :, None] * k)

            B_quats_base.append(qB_base)
            B_quats_rt.append(qB_t)
            B_gates.append(bala_gate)

            # Forward Kinematics
            localB_t = FK.run(parents, refB, qB_t)
            localB_t = (localB_t - local_mean) / local_std
            B_locals_rt.append(localB_t)

            localB_base_t = FK.run(parents, refB, qB_base)
            localB_base_t = (localB_base_t - local_mean) / local_std
            B_locals_base.append(localB_base_t)

        # stack all frames
        quatB_rt = torch.stack(B_quats_rt, dim=1)  # shape: (batch_size, T, 22, 4)
        delta_qs = torch.stack(delta_qs, dim=1)
        localB_rt = torch.stack(B_locals_rt, dim=1)  # shape: (batch_size, T, 22, 3)
        quatB_base = torch.stack(B_quats_base, dim=1)
        bala_gates = torch.stack(B_gates, dim=1)  # shape: (batch_size, T, 22)
        delta_qg = torch.stack(delta_qg, dim=1)
        localB_base = torch.stack(B_locals_base, dim=1)  # shape: (batch_size, T, 22, 3)

        """ mapping global movements from A to B"""
        globalA_vel = seqA[:, :, -4:-1]
        globalA_rot = seqA[:, :, -1]
        normalized_vin = torch.cat(
            (
                torch.divide(globalA_vel, inp_height[:, :, None]),
                globalA_rot[:, :, None],
            ),
            dim=-1,
        )
        normalized_vout = normalized_vin.clone()

        globalB_vel = normalized_vout[:, :, :-1]
        globalB_rot = normalized_vout[:, :, -1]
        globalB_rt = torch.cat(
            (
                torch.multiply(globalB_vel, tgt_height[:, :, None]),
                globalB_rot[:, :, None],
            ),
            dim=-1,
        )  # shape: (batch_size, T, 4)

        return localB_rt, globalB_rt, quatB_rt, delta_qs, delta_qg
