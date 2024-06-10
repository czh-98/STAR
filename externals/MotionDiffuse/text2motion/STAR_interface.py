import os
import torch
import numpy as np
import argparse
from os.path import join as pjoin

from utils.plot_script import *
from utils.get_opt import get_opt

from trainers import DDPMTrainer
from models import MotionTransformer
from utils.utils import *
from utils.motion_process import recover_from_ric

from utils.hybrik_loc2rot import HybrIKJointsToRotmat

from scipy.spatial.transform import Rotation as RRR


def build_models(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff,
    )
    return encoder


def motion_to_pose(data):

    joint = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()
    joint = motion_temporal_filter(joint, sigma=1)

    mgpt_data = joint.reshape(-1, 22, 3)

    mgpt_data = mgpt_data - mgpt_data[0, 0]
    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(mgpt_data)

    r = RRR.from_rotvec(np.array([0, 0, 0.0]))
    pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
    vid = []
    aroot = mgpt_data[:, 0]
    aroot[:, 1:] = -aroot[:, 1:]
    params = dict(pred_shape=np.zeros([1, 10]), pred_root=aroot, pred_pose=pose)

    pose = RRR.from_matrix(pose.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 22, 3)

    return pose


def text_to_motion(text="A person is dancing"):

    device = torch.device("cuda")
    opt = get_opt("./data/t2m/t2m_motiondiffuse/opt.txt", device)
    opt.do_denoise = True

    motion_length = 196

    opt.data_root = "./dataset/HumanML3D"
    opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
    opt.text_dir = pjoin(opt.data_root, "texts")
    opt.joints_num = 22
    opt.dim_pose = 263

    mean = np.load("./data/t2m/t2m_motiondiffuse/meta/mean.npy")
    std = np.load("./data/t2m/t2m_motiondiffuse/meta/std.npy")

    encoder = build_models(opt).to(device)
    trainer = DDPMTrainer(opt, encoder)
    trainer.load("./data/t2m/t2m_motiondiffuse/model/latest.tar")

    trainer.eval_mode()
    trainer.to(opt.device)

    with torch.no_grad():
        if motion_length != -1:
            caption = [text]
            m_lens = torch.LongTensor([motion_length]).to(device)
            pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
            motion = pred_motions[0].cpu().numpy()
            motion = motion * std + mean

            pose = motion_to_pose(motion)

    return pose
