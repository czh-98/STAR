import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
from functools import lru_cache

import pickle as pkl

import numpy as np
import json
import trimesh
from lib.common.utils import trunc_rev_sigmoid, SMPLXSeg


EMBEDDING_INDICES = [
    276,
    282,
    283,
    285,
    293,
    295,
    296,
    300,
    334,
    336,
    46,
    52,
    53,
    55,
    63,
    65,
    66,
    70,
    105,
    107,
    249,
    263,
    362,
    373,
    374,
    380,
    381,
    382,
    384,
    385,
    386,
    387,
    388,
    390,
    398,
    466,
    7,
    33,
    133,
    144,
    145,
    153,
    154,
    155,
    157,
    158,
    159,
    160,
    161,
    163,
    173,
    246,
    168,
    6,
    197,
    195,
    5,
    4,
    129,
    98,
    97,
    2,
    326,
    327,
    358,
    0,
    13,
    14,
    17,
    37,
    39,
    40,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
]

sorter = np.argsort(EMBEDDING_INDICES)
# from right to left, the upper and lower are in correspondence
UPPER_OUTTER_LIP_LINE = [185, 40, 39, 37, 0, 267, 269, 270, 409]
LOWER_OUTTER_LIP_LINE = [146, 91, 181, 84, 17, 314, 405, 321, 375]

UPPER_OUTTER_LIP_LINE_EM = sorter[
    np.searchsorted(EMBEDDING_INDICES, UPPER_OUTTER_LIP_LINE, sorter=sorter)
]
LOWER_OUTTER_LIP_LINE_EM = sorter[
    np.searchsorted(EMBEDDING_INDICES, LOWER_OUTTER_LIP_LINE, sorter=sorter)
]

# from right to left, the upper and lower are in correspondence
UPPER_INNER_LIP_LINE = [191, 80, 81, 82, 13, 312, 311, 310, 415]
LOWER_INNER_LIP_LINE = [95, 88, 178, 87, 14, 317, 402, 318, 324]

LOWER_INNER_LIP_LINE_EM = sorter[
    np.searchsorted(EMBEDDING_INDICES, LOWER_INNER_LIP_LINE, sorter=sorter)
]
UPPER_INNER_LIP_LINE_EM = sorter[
    np.searchsorted(EMBEDDING_INDICES, UPPER_INNER_LIP_LINE, sorter=sorter)
]


@lru_cache(maxsize=None)
def get_flame_vertex_idx():

    return SMPLXSeg.smplx_flame_vid


def laplacian_smooth(
    pred_v,
    faces,
):
    if len(pred_v.shape) == 2:
        pred_v = pred_v.unsqueeze(0)
    if len(faces.shape) == 2:
        faces = faces.unsqueeze(0)
    b, _, _ = pred_v.shape
    meshes = Meshes(verts=pred_v, faces=faces)
    l_matrix = meshes.laplacian_packed()

    loss = l_matrix.mm(meshes.verts_packed())

    loss = loss.norm(dim=1)

    # loss[get_flame_vertex_idx()] *= 5
    # loss = loss.sum() / b

    loss = (loss.sum() + loss[get_flame_vertex_idx()].sum() * 4) / b

    return loss


def skeleton_ranking_loss(canno_skeleton=None):

    canno_skeleton = canno_skeleton[:18].unsqueeze(0)  # 1 128 3

    # the leg length should not less than the arm length
    rarm_length = torch.norm(
        canno_skeleton[:, 3] - canno_skeleton[:, 2], dim=-1
    ) + torch.norm(canno_skeleton[:, 3] - canno_skeleton[:, 4], dim=-1)

    rleg_length = torch.norm(
        canno_skeleton[:, 9] - canno_skeleton[:, 8], dim=-1
    ) + torch.norm(canno_skeleton[:, 9] - canno_skeleton[:, 10], dim=-1)

    larm_length = torch.norm(
        canno_skeleton[:, 6] - canno_skeleton[:, 5], dim=-1
    ) + torch.norm(canno_skeleton[:, 6] - canno_skeleton[:, 7], dim=-1)

    lleg_length = torch.norm(
        canno_skeleton[:, 12] - canno_skeleton[:, 11], dim=-1
    ) + torch.norm(canno_skeleton[:, 12] - canno_skeleton[:, 13], dim=-1)

    loss = torch.max(
        torch.zeros_like(rarm_length), rarm_length - rleg_length
    ) + torch.max(torch.zeros_like(larm_length), larm_length - lleg_length)

    loss = loss.sum()

    return loss


@lru_cache(maxsize=None)
def get_lip_index():

    face_idx = SMPLXSeg.smplx_flame_vid
    head_tri = trimesh.load(
        "./data/FLAME_masks/FLAME.obj",
        maintain_order=True,
        process=False,
    ).faces

    head_tri = torch.from_numpy(head_tri).cuda()

    lmk_embeddings_mediapipe = np.load(
        "./data/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz",
        allow_pickle=True,
        encoding="latin1",
    )

    lmk_faces_idx_mediapipe = lmk_embeddings_mediapipe["lmk_face_idx"].astype(np.int64)

    lmk_bary_coords_mediapipe = lmk_embeddings_mediapipe["lmk_b_coords"]

    lmk_faces_idx_mediapipe = (
        torch.from_numpy(lmk_faces_idx_mediapipe).cuda().unsqueeze(0)
    )  # 1 105 3
    lmk_bary_coords_mediapipe = (
        torch.from_numpy(lmk_bary_coords_mediapipe).cuda().unsqueeze(0).float()
    )  # 1 105 3

    return face_idx, head_tri, lmk_faces_idx_mediapipe, lmk_bary_coords_mediapipe


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3
    )

    lmk_faces += (
        torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1)
        * num_verts
    )

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)

    landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])
    return landmarks


def lip_distance_loss(dense_mesh_vertices, dense_face=None):

    assert dense_mesh_vertices.shape[0] == 1
    assert dense_mesh_vertices.shape[1] == 25193

    # lip_landmark = None

    face_idx, face_tri, lmk_faces_idx_mediapipe, lmk_bary_coords_mediapipe = (
        get_lip_index()
    )

    vertices = dense_mesh_vertices[:, face_idx]
    landmarks2d_mediapipe = vertices2landmarks(
        vertices, face_tri, lmk_faces_idx_mediapipe, lmk_bary_coords_mediapipe
    )

    upper_lip = landmarks2d_mediapipe[:, UPPER_OUTTER_LIP_LINE_EM]

    lower_lip = landmarks2d_mediapipe[:, LOWER_OUTTER_LIP_LINE_EM]

    upper_inner_lip = landmarks2d_mediapipe[:, UPPER_INNER_LIP_LINE_EM]

    lower_inner_lip = landmarks2d_mediapipe[:, LOWER_INNER_LIP_LINE_EM]

    # y-axis
    delta = (
        upper_lip[..., 1] - lower_lip[..., 1]
    )  # the upper lip should be higher than the lower lip

    lip_dist = torch.norm(delta, dim=-1)

    # peanalize the negative distance
    distance = torch.where(delta < 0, lip_dist, torch.zeros_like(delta))

    loss = torch.sum(distance)

    # inner
    delta_inner = (
        upper_inner_lip[..., 1] - lower_inner_lip[..., 1]
    )  # the upper inner should be higher than the lower inner

    inner_dist = torch.norm(delta_inner, dim=-1)

    # peanalize the negative distance
    distance2 = torch.where(delta_inner < 0, inner_dist, torch.zeros_like(delta_inner))

    loss += torch.sum(distance2)

    up_delta_inner_lip = (
        upper_lip[..., 1] - upper_inner_lip[..., 1]
    )  # the upper lip should be higher than the upper inner

    up_inner_lip_dist = torch.norm(up_delta_inner_lip, dim=-1)

    # peanalize the negative distance
    distance3 = torch.where(
        up_delta_inner_lip < 0, up_inner_lip_dist, torch.zeros_like(up_delta_inner_lip)
    )

    loss += torch.sum(distance3)

    low_delta_inner_lip = (
        lower_inner_lip[..., 1] - lower_lip[..., 1]
    )  # the lower inner should be higher than the lower lip

    low_inner_lip_dist = torch.norm(low_delta_inner_lip, dim=-1)

    # peanalize the negative distance

    distance4 = torch.where(
        low_delta_inner_lip < 0,
        low_inner_lip_dist,
        torch.zeros_like(low_delta_inner_lip),
    )

    loss += torch.sum(distance4)

    return loss


@lru_cache(maxsize=None)
def get_eye_index():
    # NOTE: these regions do not remesh, so keep the same as initial idx

    coarse_forehead_lidx = SMPLXSeg.leye_ids
    coarse_forehead_ridx = SMPLXSeg.reye_ids
    coarse_leyeball_ids = SMPLXSeg.leyeball_ids
    coarse_reyeball_ids = SMPLXSeg.reyeball_ids

    return (
        coarse_forehead_lidx,
        coarse_forehead_ridx,
        coarse_leyeball_ids,
        coarse_reyeball_ids,
    )


def ptp(input, dim=None, keepdim=False):
    if dim is None:
        return input.max() - input.min()
    return input.max(dim, keepdim).values - input.min(dim, keepdim).values


# @lru_cache(maxsize=None)
def get_eyeball_idxs_and_rad(eye_vgroup, template_eye_vertices):

    v_idx_min_x = eye_vgroup[torch.argmin(template_eye_vertices[:, :, 0])]
    v_idx_max_x = eye_vgroup[torch.argmax(template_eye_vertices[:, :, 0])]

    eyeball_radius = torch.max(ptp(template_eye_vertices, dim=1)) / 2.0

    return v_idx_min_x, v_idx_max_x, eyeball_radius


def intersection_loss(dense_mesh_vertices):
    #
    assert dense_mesh_vertices.shape[0] == 1
    assert dense_mesh_vertices.shape[1] == 25193

    loss = torch.tensor(0).cuda()

    leyelid_idx, reyelid_idx, leyeball_idx, reyeball_idx = get_eye_index()

    # left,
    leyelid_verts = dense_mesh_vertices[:, leyelid_idx]
    reyelid_verts = dense_mesh_vertices[:, reyelid_idx]

    leyeball = dense_mesh_vertices[:, leyeball_idx]

    reyeball = dense_mesh_vertices[:, reyeball_idx]

    leye_idx_min, leye_idx_max, leyeball_radius = get_eyeball_idxs_and_rad(
        leyeball_idx, leyeball
    )

    leyeball_location = (
        dense_mesh_vertices[:, leye_idx_min] + dense_mesh_vertices[:, leye_idx_max]
    ) / 2.0

    reye_idx_min, reye_idx_max, reyeball_radius = get_eyeball_idxs_and_rad(
        reyeball_idx, reyeball
    )

    reyeball_location = (
        dense_mesh_vertices[:, reye_idx_min] + dense_mesh_vertices[:, reye_idx_max]
    ) / 2.0

    ldeltas = torch.norm(leyelid_verts - leyeball_location.unsqueeze(0), dim=-1)

    ldistance = torch.where(
        ldeltas < leyeball_radius, ldeltas, torch.zeros_like(ldeltas)
    )

    rdeltas = torch.norm(reyelid_verts - reyeball_location.unsqueeze(0), dim=-1)

    rdistance = torch.where(
        rdeltas < reyeball_radius, rdeltas, torch.zeros_like(rdeltas)
    )

    loss = loss + torch.sum(ldistance) + torch.sum(rdistance)

    return loss
