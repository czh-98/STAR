import torch
import torch.nn as nn

import numpy as np
from src.model_shape_aware import RetNet

import cv2

from os.path import exists, join
import os

import pytorch3d
import pytorch3d.transforms as tf3d
import json

import scipy.ndimage.filters as filters

import pickle as pkl
from smplx import SMPLX

import trimesh


import scipy


import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


sys.path.insert(1, os.path.join(ROOT_DIR, "outside-code"))

import BVH as BVH

import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots


sys.path.insert(2, "./externals/MotionDiffuse/text2motion/")
from STAR_interface import text_to_motion as MotionDiffuse


sys.path.insert(3, "./externals/mmhuman3d/")
from mmhuman3d.utils.demo_utils import smooth_process


def smoothness(x):
    return smooth_process(
        x,
        smooth_type="smoothnet_windowsize8",
        cfg_base_dir="./externals/mmhuman3d/configs/_base_/post_processing/",
    )


def getmodel(
    weight_path=os.path.join(ROOT_DIR, "pretrain/shape_aware.pt"),
):
    model = RetNet(
        num_joint=22,
        token_channels=64,
        hidden_channels_p=256,
        embed_channels_p=128,
        kp=1.0,
    )
    model = nn.DataParallel(model, device_ids=[0])

    print("load weight from: " + weight_path)
    weights = torch.load(weight_path)
    model.load_state_dict(weights)
    model = model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model.cuda()


def get_skel_tensor(joints, parents):
    c_offsets = []
    for j in range(parents.shape[0]):
        if parents[j] != -1:
            c_offsets.append(joints[j, :] - joints[parents[j], :])
        else:
            c_offsets.append(joints[j, :])
    return torch.stack(c_offsets, dim=0)


def get_width_tensor(vertices):
    box = torch.zeros((2, 3))
    box[0, :] = vertices.min(dim=0).values
    box[1, :] = vertices.max(dim=0).values
    width = box[1, :] - box[0, :]
    return width


def get_height_tensor(joints):
    return (
        torch.sqrt(((joints[5, :] - joints[4, :]) ** 2).sum(dim=-1))
        + torch.sqrt(((joints[4, :] - joints[3, :]) ** 2).sum(dim=-1))
        + torch.sqrt(((joints[3, :] - joints[2, :]) ** 2).sum(dim=-1))
        + torch.sqrt(((joints[2, :] - joints[1, :]) ** 2).sum(dim=-1))
        + torch.sqrt(((joints[1, :] - joints[0, :]) ** 2).sum(dim=-1))
        + torch.sqrt(((joints[6, :] - joints[7, :]) ** 2).sum(dim=-1))
        + torch.sqrt(((joints[7, :] - joints[8, :]) ** 2).sum(dim=-1))
        + torch.sqrt(((joints[8, :] - joints[9, :]) ** 2).sum(dim=-1))
    )


def transforms(quat):

    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    shape = quat.shape[:-1]
    m = torch.empty(shape + (3, 3)).cuda()

    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


def softmax_tensor(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = torch.max(x, **kw).values, torch.min(x, **kw).values
    return maxi + torch.log(softness + torch.exp(mini - maxi))


def softmin_tensor(x, **kw):
    return -softmax_tensor(-x, **kw)


def between(v0s, v1s):
    a = torch.cross(v0s, v1s)
    w = torch.sqrt((v0s**2).sum(dim=-1) * (v1s**2).sum(dim=-1)) + (v0s * v1s).sum(dim=-1)

    result = torch.concat([w[..., None], a], dim=-1)

    length = torch.sum(result**2.0, dim=-1) ** 0.5

    return result / length[..., None]


def get_orient_start_tensor(reference, sdr_l, sdr_r, hip_l, hip_r):
    """Get Forward Direction"""
    across1 = reference[0:1, hip_l] - reference[0:1, hip_r]
    across0 = reference[0:1, sdr_l] - reference[0:1, sdr_r]
    across = across0 + across1
    across = across / torch.sqrt((across**2).sum(dim=-1))[..., None]

    direction_filterwidth = 20
    forward = torch.cross(across, torch.from_numpy(np.array([[0.0, 1.0, 0.0]])).float().cuda())

    forward = filters.gaussian_filter1d(forward.detach().cpu().numpy(), direction_filterwidth, axis=0, mode="nearest")
    forward = torch.from_numpy(forward).cuda()

    forward = forward / torch.sqrt((forward**2).sum(dim=-1))[..., None]
    """ Add Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    target = torch.from_numpy(target).float().cuda()

    rotation = between(forward, target)[:, None]

    return -rotation


def put_in_world_bvh_tensor(states, start_rots):
    joints = states[:, :-4]
    root_x = states[:, -4]
    root_y = states[:, -3]
    root_z = states[:, -2]
    root_r = states[:, -1]

    joints = joints.reshape(joints.shape[:1] + (-1, 3))

    qs = np.zeros((1, 4))
    qs[:, 0] = 1.0
    qs = torch.from_numpy(qs).cuda()

    rotation = tf3d.quaternion_raw_multiply(start_rots[0], qs)

    rotations = []
    offsets = []

    translation = np.array([[0.0, 0.0, 0.0]])
    translation = torch.from_numpy(translation).float().cuda()

    for i in range(len(joints)):
        joints[i, 1:, :] = tf3d.quaternion_apply(rotation, joints[i, 1:, :])
        joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
        joints[i, :, 1] = joints[i, :, 1] + translation[0, 1]
        joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]

        rotations.append(rotation[:, None, :])

        axis = torch.from_numpy(np.array([0, 1.0, 0])).cuda()
        axis = axis / (torch.sqrt(torch.sum(axis**2, dim=-1)) + 1e-10)[..., None]
        sines = torch.sin(-root_r[i] / 2.0)[..., None]
        cosines = torch.cos(-root_r[i] / 2.0)[..., None]

        rotation = tf3d.quaternion_raw_multiply(
            torch.concat([cosines, axis * sines], dim=-1).unsqueeze(0).float(), rotation
        )

        offsets.append(tf3d.quaternion_apply(rotation, torch.from_numpy(np.array([0.0, 0, 1.0])).float().cuda()))

        translation = translation + tf3d.quaternion_apply(
            rotation,
            torch.tensor([root_x[i], root_y[i], root_z[i]]).float().cuda(),
        )

    return joints[None], torch.concat(rotations, dim=0)


def process_tensor(positions):
    """Put on Floor"""

    fid_l, fid_r = np.array([8, 9]), np.array([12, 13])
    fid_l = torch.from_numpy(fid_l)
    fid_r = torch.from_numpy(fid_r)

    foot_heights = torch.minimum(positions[:, fid_l, 1], positions[:, fid_r, 1]).min(dim=1).values
    floor_height = softmin_tensor(foot_heights, softness=0.5, dim=0)

    positions[:, :, 1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = positions[:, 0]
    positions = torch.concat([reference[:, np.newaxis], positions], dim=1)

    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.15, 0.15]), np.array([9.0, 6.0])

    velfactor = torch.from_numpy(velfactor).cuda()
    heightfactor = torch.from_numpy(heightfactor).cuda()

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).float()

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).float()

    """ Get Root Velocity """
    velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).clone()

    """ Remove Translation """
    positions[:, :, 0] = positions[:, :, 0] - positions[:, :1, 0]
    positions[1:, 1:, 1] = positions[1:, 1:, 1] - (positions[1:, :1, 1] - positions[:1, :1, 1])
    positions[:, :, 2] = positions[:, :, 2] - positions[:, :1, 2]

    """ Get Forward Direction """
    # Original indices + 1 for added reference joint
    sdr_l, sdr_r, hip_l, hip_r = 15, 19, 7, 11
    across1 = positions[:, hip_l] - positions[:, hip_r]
    across0 = positions[:, sdr_l] - positions[:, sdr_r]
    across = across0 + across1
    across = across / torch.sqrt((across**2).sum(axis=-1))[..., None]

    direction_filterwidth = 20
    forward = torch.cross(across, torch.from_numpy(np.array([[0.0, 1.0, 0.0]])).float().cuda())

    #############
    forward = filters.gaussian_filter1d(forward.detach().cpu().numpy(), direction_filterwidth, axis=0, mode="nearest")

    forward = torch.from_numpy(forward).cuda()
    forward = forward / torch.sqrt((forward**2).sum(dim=-1))[..., None]

    """ Remove Y Rotation """
    target = torch.from_numpy(np.array([[0, 0, 1]]).repeat(len(forward), axis=0)).float().cuda()

    rotation = between(forward, target)[:, None]

    positions = tf3d.quaternion_apply(rotation, positions)

    """ Get Root Rotation """
    velocity = tf3d.quaternion_apply(rotation[1:], velocity)

    rvelocity = tf3d.quaternion_to_axis_angle(
        tf3d.quaternion_multiply(rotation[1:], tf3d.quaternion_invert(rotation[:-1]))
    )[..., 1]

    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)

    positions = torch.concat([positions, velocity[:, :, 0]], dim=-1)
    positions = torch.concat([positions, velocity[:, :, 1]], dim=-1)
    positions = torch.concat([positions, velocity[:, :, 2]], dim=-1)
    positions = torch.concat([positions, rvelocity], dim=-1)
    positions = torch.concat([positions, feet_l, feet_r], dim=-1)

    return positions, rotation


def transforms_blank(rotations):
    shape = (rotations.shape[0], rotations.shape[1])
    ts = torch.zeros(shape + (4, 4)).cuda()
    ts[:, :, 0, 0] = 1.0
    ts[:, :, 1, 1] = 1.0
    ts[:, :, 2, 2] = 1.0
    ts[:, :, 3, 3] = 1.0

    return ts


def transforms_local(rotations, positions):
    transform = transforms(rotations)
    transform = torch.concat([transform, torch.zeros(transform.shape[:2] + (3, 1)).cuda()], dim=-1)
    transform = torch.concat([transform, torch.zeros(transform.shape[:2] + (1, 4)).cuda()], dim=-2)
    transform[:, :, 0:3, 3] = positions
    transform[:, :, 3:4, 3] = 1.0

    return transform


def transforms_multiply(t0s, t1s):
    return t0s @ t1s


def transforms_global(
    rotations,
    positions,
    parents,
):
    locals = transforms_local(rotations, positions)  # y
    globals = transforms_blank(rotations)  # y

    globals[:, 0] = locals[:, 0]

    for i in range(1, rotations.shape[1]):
        globals[:, i] = transforms_multiply(globals[:, parents[i]], locals[:, i])
    return globals


def positions_global(
    rotations,
    positions,
    parents,
):
    positions = transforms_global(
        rotations,
        positions,
        parents,
    )[:, :, :, 3]
    return positions[:, :, :3] / positions[:, :, 3, None]


class JointMapper(nn.Module):
    """
    # https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py
    """

    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer("joint_maps", torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


class motion_retarget(nn.Module):
    def __init__(
        self,
        stats_path=os.path.join(ROOT_DIR, "datasets/mixamo/stats"),
        text_prompt="",
        t2m_model="mdiffuse",
        smpl_path=None,
        smpl_seg_path=None,
    ) -> None:
        super().__init__()

        self.parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20])
        self.retarget_net = getmodel()

        self.joints_list = [
            "Spine",
            "Spine1",
            "Spine2",
            "Neck",
            "Head",
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
            "LeftToeBase",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
            "RightToeBase",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
        ]

        self.bvh_to_smpl = {
            0: 0,  # hips -> MidHip # "Hips",
            1: 3,  # spine -> spine # "Spine",
            2: 6,  # spine1 -> spine1 # "Spine1",
            3: 9,  # spine2 -> spine2 # "Spine2",
            4: 12,  # neck -> neck # "Neck",
            5: 15,  # head ??? # "Head",
            6: 1,  # leftUpLeg -> leftUpLeg # "LeftUpLeg",
            7: 4,  # leftLeg -> leftLeg # "LeftLeg",
            8: 7,  # leftFoot -> leftFoot # "LeftFoot",
            9: 10,  # leftToeBase # "LeftToeBase",
            10: 2,  # rightUpLeg -> rightUpLeg # "RightUpLeg",
            11: 5,  # rightLeg -> rightLeg # "RightLeg",
            12: 8,  # rightFoot -> rightFoot # "RightFoot",
            13: 11,  # rightToeBase -> rightToeBase # "RightToeBase",
            14: 13,  # leftShoulder -> leftShoulder # "LeftShoulder",
            15: 16,  # leftArm -> leftArm # "LeftArm",
            16: 18,  # leftForeArm -> leftForeArm # "LeftForeArm",
            17: 20,  # leftHand -> leftHand # "LeftHand",
            18: 14,  # rightShoulder -> rightShoulder # "RightShoulder",
            19: 17,  # rightArm -> rightArm # "RightArm",
            20: 19,  # rightForeArm -> rightForeArm # "RightForeArm",
            21: 21,  # rightHand -> rightHand # "RightHand",
        }

        self.smpl_to_bvh = dict([val, key] for key, val in self.bvh_to_smpl.items())

        self.bvh_to_smpl_idx = [self.bvh_to_smpl[i] for i in range(22)]
        self.smpl_to_bvh_idx = [self.smpl_to_bvh[i] for i in range(22)]

        smpl = SMPLX(
            smpl_path,
            batch_size=1,
            num_betas=300,
            create_betas=True,
            joint_mapper=JointMapper(torch.LongTensor(self.bvh_to_smpl_idx)),
            flat_hand_mean=False,
            # flat_hand_mean=True,
        )

        self.smpl = smpl.cuda()

        with open(smpl_seg_path) as f:
            data = json.load(f)

        self.left_hand_index = data["leftArm"] + data["leftHand"] + data["leftHandIndex1"] + data["leftForeArm"]

        self.body_index = (
            data["spine1"]
            + data["spine2"]
            + data["leftShoulder"]
            + data["rightShoulder"]
            + data["neck"]
            + data["spine"]
            + data["hips"]
        )

        self.smpl_segment_map = data

        ### load stats
        local_mean = np.load(join(stats_path, "mixamo_local_motion_mean.npy"))
        local_std = np.load(join(stats_path, "mixamo_local_motion_std.npy"))
        global_mean = np.load(join(stats_path, "mixamo_global_motion_mean.npy"))
        global_std = np.load(join(stats_path, "mixamo_global_motion_std.npy"))
        quat_mean = np.load(join(stats_path, "mixamo_quat_mean.npy"))
        quat_std = np.load(join(stats_path, "mixamo_quat_std.npy"))
        local_std[local_std == 0] = 1

        self.local_mean = local_mean
        self.local_std = local_std
        self.global_mean = global_mean
        self.global_std = global_std
        self.quat_mean = quat_mean
        self.quat_std = quat_std

        self.local_mean_tensor = torch.from_numpy(local_mean.astype(np.single)).cuda()
        self.local_std_tensor = torch.from_numpy(local_std.astype(np.single)).cuda()
        self.global_mean_tensor = torch.from_numpy(global_mean.astype(np.single)).cuda()
        self.global_std_tensor = torch.from_numpy(global_std.astype(np.single)).cuda()
        self.quat_mean_tensor = torch.from_numpy(quat_mean.astype(np.single)).cuda()
        self.quat_std_tensor = torch.from_numpy(quat_std.astype(np.single)).cuda()

        if t2m_model == "mdiffuse":
            init_motion = MotionDiffuse(text_prompt)

        self.init_motion = torch.from_numpy(init_motion).cuda()

        #############
        (
            self.inp_seq,
            self.inpskel,
            self.inpquat,
            self.inp_shape,
            self.inp_height_,
        ) = self.preprocess_smpl_input(motion=self.init_motion)

    def preprocess_smpl_input(self, motion=None):

        beta = torch.zeros(1, 300).cuda()

        (
            inp_anim,
            inpskel,
            inseq,
            inpquat,
            inp_full_width,
            inp_joint_shape,
            source_position,
            source_rotation,
        ) = self.get_inp_from_SMPL_pose(beta)

        offset = inseq[:, -8:-4]
        inseq = torch.reshape(inseq[:, :-8], [inseq.shape[0], -1, 3])

        inp_shape = torch.divide(inp_joint_shape, inp_full_width[None, :]).reshape(-1)
        inp_skel = inpskel[0, :].reshape([22, 3])

        inp_height = get_height_tensor(inp_skel) / 100

        inpskel = (inpskel - self.local_mean_tensor) / self.local_std_tensor
        inpskel = inpskel.reshape([inpskel.shape[0], -1])

        inpskel = inpskel[None, :].float()
        inp_shape = inp_shape[None, :].float()
        inp_height_ = torch.zeros((1, 1)).cuda()
        inp_height_[0, 0] = inp_height

        #
        inpquat = (inpquat - self.quat_mean_tensor) / self.quat_std_tensor

        inseq = (inseq - self.local_mean_tensor) / self.local_std_tensor

        inseq = inseq.reshape([inseq.shape[0], -1])

        inp_seq = torch.concat((inseq, offset), dim=-1)

        return (
            inp_seq.unsqueeze(0).float(),
            inpskel,
            inpquat.unsqueeze(0).float(),
            inp_shape,
            inp_height_,
        )

    def forward(self, beta=torch.ones(1, 300).cuda()):

        (
            smpl_anim,
            tgtskel,
            tgtseq,
            tgtquat,
            tgt_full_width,
            tgt_joint_shape,
            reference_position,
            reference_rotation,
        ) = self.get_inp_from_SMPL_pose(beta)

        #### load source data ####
        inp_seq = self.inp_seq
        inpskel = self.inpskel
        inpquat = self.inpquat
        inp_shape = self.inp_shape
        inp_height_ = self.inp_height_

        T = inpskel.shape[0]

        ### prepare SMPL target shape ####
        tgt_shape = torch.divide(tgt_joint_shape, tgt_full_width[None, :]).reshape(-1)

        out_skel = tgtskel[0, :].reshape([22, 3])

        tgt_height = get_height_tensor(out_skel) / 100

        tgtskel = (tgtskel - self.local_mean_tensor) / self.local_std_tensor
        tgtskel = tgtskel.reshape([tgtskel.shape[0], -1])

        tgtskel = tgtskel[None, :].float()
        tgt_shape = tgt_shape[None, :].float()
        tgt_height_ = torch.zeros((1, 1)).cuda()
        tgt_height_[0, 0] = tgt_height

        oursL, oursG, quatsB, delta_q, delta_s = self.retarget_net(
            inp_seq,
            None,
            inpskel,
            tgtskel,
            inp_shape,
            tgt_shape,
            inpquat,
            inp_height_,
            tgt_height_,
            self.local_mean_tensor,
            self.local_std_tensor,
            self.quat_mean_tensor,
            self.quat_std_tensor,
            self.parents,
            0.8,
        )

        localB = oursL.clone()

        oursL = oursL.reshape([oursL.shape[0], oursL.shape[1], -1])
        local_mean_rshp = self.local_mean_tensor.reshape((1, 1, -1))
        local_std_rshp = self.local_std_tensor.reshape((1, 1, -1))
        oursL[:, :, :] = oursL[:, :, :] * local_std_rshp + local_mean_rshp
        oursG[:, :, :] = oursG[:, :, :]

        localB = localB * self.local_std_tensor[None, :] + self.local_mean_tensor[None, :]

        ours_total = torch.concat((oursL, oursG), dim=-1)

        ####
        max_steps = tgtskel.shape[1]
        tjoints = torch.reshape(tgtskel[0] * local_std_rshp + local_mean_rshp, [max_steps, -1, 3])

        outputB_bvh = ours_total[0].clone()

        """Follow the same motion direction as the input and zero speeds that are zero in the input."""
        outputB_bvh[:, -4:] = outputB_bvh[:, -4:] * (torch.sign(inp_seq[0, :, -4:]) * torch.sign(ours_total[0, :, -4:]))
        outputB_bvh[:, -3][torch.abs(inp_seq[0, :, -3]) <= 1e-2] = 0.0

        outputB_bvh[:, :3] = reference_position[:1, 0, :].clone()

        tmp_gt = positions_global(
            reference_rotation,
            reference_position,
            smpl_anim.parents,
        )

        tgtjoints = np.arange(22)

        start_rots = get_orient_start_tensor(
            tmp_gt,
            tgtjoints[14],  # left shoulder
            tgtjoints[18],  # right shoulder
            tgtjoints[6],  # left upleg
            tgtjoints[10],  # right upleg
        )

        wjs, rots = put_in_world_bvh_tensor(
            outputB_bvh.clone(),
            start_rots,
        )

        tjoints[:, 0, :] = wjs[0, :, 0].clone()

        """ Quaternion results """

        cquat = quatsB[0].clone()
        cquat[:, 0:1, :] = tf3d.quaternion_multiply(rots, cquat[:, 0:1, :])
        rot_mats = transforms(cquat)

        smpl_pose = rot_mats[:, self.smpl_to_bvh_idx, :]

        smpl_pose_rotvec = tf3d.matrix_to_axis_angle(smpl_pose.reshape(-1, 3, 3)).reshape(-1, 22, 3)

        smpl_pose_rotvec = torch.from_numpy(smoothness(smpl_pose_rotvec.detach().cpu().numpy())).cuda()

        return smpl_pose_rotvec

    def get_inp_from_SMPL_pose(self, beta=torch.zeros(1, 300)):
        # template anim
        parents = self.parents

        orients = Quaternions.id(0)
        for _ in range(len(parents)):
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)

        rotations = np.zeros((1, len(orients), 3))
        rotations = Quaternions.from_euler(np.radians(rotations), order="xyz", world=False)

        if self.init_motion is not None:
            rotations.qs = (
                tf3d.matrix_to_quaternion(tf3d.axis_angle_to_matrix(self.init_motion[:, self.bvh_to_smpl_idx, :]))
                .cpu()
                .numpy()
            )

        rot_mats = transforms(torch.from_numpy(rotations.qs))

        smpl_pose = rot_mats[:, self.smpl_to_bvh_idx, :].float()

        smpl_pose_rotvec = tf3d.matrix_to_axis_angle(smpl_pose.reshape(-1, 3, 3)).reshape(-1, 22, 3)

        template = self.smpl.forward(
            betas=beta,
        )

        #### get shape information for retarge #####
        v = template.vertices.reshape(-1, 3)
        skeleton = template.joints.reshape(-1, 3)[:22]

        v = (v - skeleton[0]) * 100
        skeleton = (skeleton - skeleton[0]) * 100

        rest_vertices_data = v
        rest_arm_vertices_data = v[self.left_hand_index, :]
        rest_body_vertices_data = v[self.body_index, :]

        body_width = get_width_tensor(rest_body_vertices_data)
        full_width = get_width_tensor(rest_vertices_data)

        shape_lst = [
            get_width_tensor(v[self.smpl_segment_map["hips"], :]),
            get_width_tensor(v[self.smpl_segment_map["spine"], :]),
            get_width_tensor(v[self.smpl_segment_map["spine1"], :]),
            get_width_tensor(v[self.smpl_segment_map["spine2"], :]),
            get_width_tensor(v[self.smpl_segment_map["neck"], :]),
            get_width_tensor(v[self.smpl_segment_map["head"], :]),
            get_width_tensor(v[self.smpl_segment_map["leftUpLeg"], :]),
            get_width_tensor(v[self.smpl_segment_map["leftLeg"], :]),
            get_width_tensor(v[self.smpl_segment_map["leftFoot"], :]),
            get_width_tensor(v[self.smpl_segment_map["leftToeBase"], :]),
            get_width_tensor(v[self.smpl_segment_map["rightUpLeg"], :]),
            get_width_tensor(v[self.smpl_segment_map["rightLeg"], :]),
            get_width_tensor(v[self.smpl_segment_map["rightFoot"], :]),
            get_width_tensor(v[self.smpl_segment_map["rightToeBase"], :]),
            get_width_tensor(v[self.smpl_segment_map["leftShoulder"], :]),
            get_width_tensor(v[self.smpl_segment_map["leftArm"], :]),
            get_width_tensor(v[self.smpl_segment_map["leftForeArm"], :]),
            get_width_tensor(v[self.smpl_segment_map["leftHand"], :]),
            get_width_tensor(v[self.smpl_segment_map["rightShoulder"], :]),
            get_width_tensor(v[self.smpl_segment_map["rightArm"], :]),
            get_width_tensor(v[self.smpl_segment_map["rightForeArm"], :]),
            get_width_tensor(v[self.smpl_segment_map["rightHand"], :]),
        ]
        shape_lst_array = torch.stack(shape_lst, dim=0)

        tgt_full_width = full_width.float()
        tgt_joint_shape = shape_lst_array.float()

        joints = template.joints.repeat(smpl_pose_rotvec.shape[0], 1, 1) * 100

        positions = joints.clone()

        positions = positions - positions[0][0]

        positions[:, 1:, :] = positions[:, 1:, :] - positions[:, parents[1:], :]

        offsets = positions[0]

        smpl_anim = Animation.Animation(rotations, positions, orients, offsets, parents)

        reference_rot = torch.from_numpy(rotations.qs).cuda()
        reference_orient = torch.from_numpy(smpl_anim.orients.qs).cuda()

        joints = positions_global(
            reference_rot,
            positions,
            parents,
        )

        joints = torch.concat([joints, joints[-1:]], dim=0)

        new_joints, rotation = process_tensor(joints)

        new_joints = new_joints[:, 3:]

        rotation = rotation[:-1]

        data_seq = new_joints

        angle = tf3d.quaternion_multiply(rotation[:, 0, :], reference_rot[:, 0, :]).float()

        smpl_anim.rotations.qs[:, 0, :] = angle.detach().cpu().numpy()

        reference_rot[:, 0, :] = angle

        data_quat = reference_rot.clone()

        smpl_anim.rotations.qs[...] = smpl_anim.orients.qs[None]
        reference_rot[...] = reference_orient[None]

        tjoints = positions_global(
            reference_rot,
            positions,
            smpl_anim.parents,
        )

        smpl_anim.positions[...] = get_skel_tensor(tjoints[0], smpl_anim.parents)[None]
        smpl_anim.positions[:, 0, :] = new_joints[:, :3]
        data_skel = smpl_anim.positions

        reference_position = get_skel_tensor(tjoints[0], smpl_anim.parents)[None].repeat(positions.shape[0], 1, 1)

        reference_position[:, 0, :] = new_joints[:, :3]

        return (
            smpl_anim,
            data_skel,
            data_seq,
            data_quat,
            tgt_full_width,
            tgt_joint_shape,
            reference_position,
            reference_rot,
        )
