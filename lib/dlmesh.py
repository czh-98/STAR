import os
import random

import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
from lib.common.obj import (
    Mesh,
    safe_normalize,
    normalize_vert,
    save_obj_mesh,
    compute_normal,
)
from lib.common.utils import trunc_rev_sigmoid, SMPLXSeg
from lib.common.renderer import Renderer
from lib.common.remesh import smplx_remesh_mask, subdivide, subdivide_inorder
from lib.common.lbs import warp_points

from tada_smplx import smplx

import matplotlib
import math
from PIL import Image

import sys

sys.path.insert(1, "./externals/R2ET")

from retarget_interface import motion_retarget


def is_nan(val):
    return val is None or np.isnan(val)


def draw_bodypose(canvas, keypoints_2d):
    """
    canvas = np.zeros_like(input_image), np.array, [H x W x 3]
    keypoints_2d: np.array, [N, 18, 2], N is the number of people
    """

    stickwidth = 4
    limbSeq = [
        [2, 3],  #
        [2, 6],  #
        [3, 4],  #
        [4, 5],  #
        [6, 7],  #
        [7, 8],  #
        [2, 9],  #
        [9, 10],  #
        [10, 11],  #
        [2, 12],  #
        [12, 13],  #
        [13, 14],  #
        [2, 1],  #
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        # [3, 17],
        # [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    # colors = np.array(colors)[:,::-1].astype(np.uint8).tolist()
    assert keypoints_2d.shape[1] == 18 and keypoints_2d.ndim in (2, 3)
    if keypoints_2d.ndim == 2:
        keypoints_2d = keypoints_2d[np.newaxis, ...]
    N = keypoints_2d.shape[0]

    # the order of left and right is reversed
    ridx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]

    keypoints_2d = keypoints_2d[:, ridx, :]

    for p in range(N):
        # draw points
        for i in range(18):
            x, y = keypoints_2d[p, i]
            if is_nan(x) or is_nan(y):
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
        # draw lines
        for i in range(17):
            indices = np.array(limbSeq[i]) - 1
            cur_canvas = canvas.copy()
            X = keypoints_2d[p, indices, 1]
            Y = keypoints_2d[p, indices, 0]
            if is_nan(Y[0]) or is_nan(Y[1]) or is_nan(X[0]) or is_nan(X[1]):
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas
    # return Image.fromarray(canvas)


def smpl_to_openpose(
    model_type="smplx",
    openpose_format="coco18",
    use_hands=True,
    use_face=True,
    use_face_contour=False,
):

    # coco18_names = [
    #     'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
    #     'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle',
    #     'left_hip', 'left_knee', 'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear',
    # ]

    body_mapping = np.array(
        [55, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59],
        dtype=np.int32,
    )
    mapping = [body_mapping]
    if use_hands:
        lhand_mapping = np.array(
            [
                20,
                37,
                38,
                39,
                60,
                25,
                26,
                27,
                61,
                28,
                29,
                30,
                62,
                34,
                35,
                36,
                63,
                31,
                32,
                33,
                64,
            ],
            dtype=np.int32,
        )
        rhand_mapping = np.array(
            [
                21,
                52,
                53,
                54,
                65,
                40,
                41,
                42,
                66,
                43,
                44,
                45,
                67,
                49,
                50,
                51,
                68,
                46,
                47,
                48,
                69,
            ],
            dtype=np.int32,
        )
        mapping += [lhand_mapping, rhand_mapping]
    if use_face:
        face_mapping = np.arange(70, 70 + 51 + 17 * use_face_contour, dtype=np.int32)
        mapping += [face_mapping]

    return np.concatenate(mapping)


class DLMesh(nn.Module):
    def __init__(
        self,
        opt,
        text_prompt="",
        t2m_model="mdiffuse",
    ):

        super(DLMesh, self).__init__()

        self.opt = opt

        self.num_remeshing = 1

        self.renderer = Renderer()

        self.device = torch.device("cuda")
        self.lock_beta = opt.lock_beta

        self.face_indices = (0, 14, 15, 16, 17)
        self.body_indices = [i for i in range(18) if i not in (0, 14, 15, 16, 17)]
        self.face_keypoints = ("nose", "r_eye", "l_eye", "r_ear", "l_ear")

        self.body_model = smplx.create(
            model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz",
            model_type="smplx",
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_expression=True,
            create_transl=False,
            use_pca=False,
            use_face_contour=True,
            flat_hand_mean=True,
            num_betas=300,
            num_expression_coeffs=100,
            num_pca_comps=12,
            dtype=torch.float32,
            batch_size=1,
        ).to(self.device)

        self.smplx_faces = self.body_model.faces.astype(np.int32)

        param_file = "./data/init_body/fit_smplx_params.npz"
        smplx_params = dict(np.load(param_file))
        self.betas = torch.as_tensor(smplx_params["betas"]).to(self.device)
        self.jaw_pose = torch.as_tensor(smplx_params["jaw_pose"]).to(self.device)
        self.body_pose = torch.as_tensor(smplx_params["body_pose"]).to(self.device)
        self.body_pose = self.body_pose.view(-1, 3)
        self.body_pose[[0, 1, 3, 4, 6, 7], :2] *= 0
        self.body_pose = self.body_pose.view(1, -1)

        self.global_orient = torch.as_tensor(smplx_params["global_orient"]).to(
            self.device
        )
        self.expression = torch.zeros(1, 100).to(self.device)

        self.remesh_mask = self.get_remesh_mask()
        self.faces_list, self.dense_lbs_weights, self.uniques, self.vt, self.ft = (
            self.get_init_body()
        )

        N = self.dense_lbs_weights.shape[0]

        self.mlp_texture = None

        res = self.opt.albedo_res
        albedo = torch.ones((res, res, 3), dtype=torch.float32) * 0.5  # default color
        self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(albedo))

        # Geometry parameters
        # displacement
        self.v_offsets = nn.Parameter(torch.zeros(N, 1))

        # shape
        self.betas = nn.Parameter(self.betas)

        # expression
        rich_data = np.load("./data/talkshow/rich.npy")
        self.rich_params = torch.as_tensor(
            rich_data, dtype=torch.float32, device=self.device
        )  # 1800 265

        if not self.opt.lock_expression:
            self.expression = nn.Parameter(self.expression)

        if not self.opt.lock_pose:
            self.body_pose = nn.Parameter(self.body_pose)

        self.jaw_pose = nn.Parameter(self.jaw_pose)

        self.motion_retarget = motion_retarget(
            text_prompt=text_prompt,
            t2m_model=t2m_model,
            smpl_path="./data/smplx/SMPLX_NEUTRAL_2020.npz",
            smpl_seg_path="./data/smplx/smplx_vert_segementation.json",
        )

        self.retarget_pose = None

        self.init_motion = self.motion_retarget.init_motion.float().clone()

        self.smplx_face_vertix_idx = SMPLXSeg.smplx_flame_vid

        self.text_prompt = text_prompt

    @torch.no_grad()
    def get_init_body(self, cache_path="./data/init_body/data.npz"):
        cache_path = "./data/init_body/data.npz"
        data = np.load(cache_path)
        faces_list = [torch.as_tensor(data["dense_faces"], device=self.device)]
        dense_lbs_weights = torch.as_tensor(
            data["dense_lbs_weights"], device=self.device
        )
        unique_list = [data["unique"]]
        vt = torch.as_tensor(data["vt"], device=self.device)
        ft = torch.as_tensor(data["ft"], device=self.device)

        return faces_list, dense_lbs_weights, unique_list, vt, ft

    def get_remesh_mask(self):
        ids = list(set(SMPLXSeg.front_face_ids) - set(SMPLXSeg.forehead_ids))
        ids = ids + SMPLXSeg.ears_ids + SMPLXSeg.eyeball_ids + SMPLXSeg.hands_ids
        mask = ~np.isin(np.arange(10475), ids)
        mask = mask[self.body_model.faces].all(axis=1)
        return mask

    def get_params(self, lr):
        params = []

        if not self.opt.lock_tex:
            params.append({"params": self.raw_albedo, "lr": lr * 10})

        if not self.opt.lock_geo:
            params.append({"params": self.v_offsets, "lr": 0.0001})

            if not self.lock_beta:
                params.append({"params": self.betas, "lr": 0.1})

            if not self.opt.lock_expression:
                params.append({"params": self.expression, "lr": 0.05})

            if not self.opt.lock_pose:
                params.append({"params": self.body_pose, "lr": 0.05})
                params.append({"params": self.jaw_pose, "lr": 0.05})

        return params

    def get_vertex_offset(self, is_train):
        v_offsets = self.v_offsets
        if not is_train and self.opt.replace_hands_eyes:
            v_offsets[SMPLXSeg.eyeball_ids] = 0.0
            v_offsets[SMPLXSeg.hands_ids] = 0.0
        return v_offsets

    def get_mesh(
        self,
        is_train,
        pose_seq=None,
        test_pose=None,
        sample_cano=False,
        is_full_body=True,
        # for sequence,
        return_scale=False,
        center=None,
        scale=None,
        return_skel=False,
    ):

        # if True:
        if not is_train and test_pose is not None:

            output = self.body_model(
                betas=self.betas,
                global_orient=test_pose[:, :1],
                body_pose=test_pose[:, 1:],
                jaw_pose=self.jaw_pose,
                expression=self.expression,
                return_verts=True,
            )

        elif not is_train or pose_seq is None:
            # sample canonical pose
            output = self.body_model(
                betas=self.betas,
                body_pose=self.body_pose,
                jaw_pose=self.jaw_pose,
                expression=self.expression,
                return_verts=True,
            )

        else:

            if sample_cano and random.random() < 0.1:
                output = self.body_model(
                    betas=self.betas,
                    body_pose=self.body_pose,
                    jaw_pose=self.jaw_pose,
                    expression=self.expression,
                    return_verts=True,
                )

            else:
                if is_full_body:
                    random_pose = pose_seq[
                        random.randint(0, pose_seq.shape[0] - 1)
                    ].unsqueeze(0)

                    output = self.body_model(
                        betas=self.betas,
                        body_pose=random_pose[:, 1:],
                        jaw_pose=random.choice(self.rich_params)[None, :3],
                        expression=self.expression,
                        return_verts=True,
                    )
                else:
                    # only face, use cano pose with random jaw
                    output = self.body_model(
                        betas=self.betas,
                        body_pose=self.body_pose,
                        jaw_pose=random.choice(self.rich_params)[None, :3],
                        expression=self.expression,
                        return_verts=True,
                    )

        v_cano = output.v_posed[0]
        landmarks = output.joints[0, -68:, :]

        # re-mesh
        v_cano_dense = subdivide_inorder(
            v_cano, self.smplx_faces[self.remesh_mask], self.uniques[0]
        )

        for unique, faces in zip(self.uniques[1:], self.faces_list[:-1]):
            v_cano_dense = subdivide_inorder(v_cano_dense, faces, unique)

        # add offset before warp
        if not self.opt.lock_geo:
            if self.v_offsets.shape[1] == 1:
                vn = compute_normal(v_cano_dense, self.faces_list[-1])[0]
                v_cano_dense += self.get_vertex_offset(is_train) * vn
            else:
                v_cano_dense += self.get_vertex_offset(is_train)
        # LBS
        v_posed_dense = warp_points(
            v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55]
        ).squeeze(0)

        # if not is_train:
        if center is None:
            v_posed_dense, center, scale = normalize_vert(v_posed_dense, return_cs=True)

        else:
            # use center/scale of canonical pose
            v_posed_dense = (v_posed_dense - center) * scale

        # normalize v_posed, joints, and landmarks
        v_posed = (output.v_posed[0] - center) * scale

        # normalize joints
        joints = (output.joints[0] - center) * scale
        landmarks = (landmarks - center) * scale

        mesh = Mesh(v_posed_dense, self.faces_list[-1].int(), vt=self.vt, ft=self.ft)
        mesh.auto_normal()

        if not self.opt.lock_tex and not self.opt.tex_mlp:
            mesh.set_albedo(self.raw_albedo)

        keypoints = joints[
            ...,
            smpl_to_openpose("smplx", openpose_format="coco18", use_face_contour=True),
            :,
        ]
        if return_scale:
            return mesh, landmarks, keypoints, v_posed, center, scale

        if return_skel:
            joints_cano = self.body_model(
                betas=self.betas,
                return_verts=True,
            ).joints[0]
            return mesh, landmarks, keypoints, v_posed, joints_cano, v_cano_dense
        else:
            return mesh, landmarks, keypoints, v_posed

    @torch.no_grad()
    def get_mesh_preprocess(
        self,
    ):
        output = self.body_model(
            betas=torch.zeros_like(self.betas).cuda(),
            body_pose=self.body_pose,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            return_verts=True,
        )

        v_cano = output.v_posed[0]
        landmarks = output.joints[0, -68:, :]

        # re-mesh
        v_cano_dense = subdivide_inorder(
            v_cano, self.smplx_faces[self.remesh_mask], self.uniques[0]
        )

        for unique, faces in zip(self.uniques[1:], self.faces_list[:-1]):
            v_cano_dense = subdivide_inorder(v_cano_dense, faces, unique)

        return v_cano, v_cano_dense

    @torch.no_grad()
    def get_mesh_center_scale(self, phrase):
        assert phrase in ["face", "body"]
        vertices = self.body_model(
            betas=self.betas,
            body_pose=self.body_pose,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            return_verts=True,
        ).vertices[0]
        vertices = normalize_vert(vertices)

        if phrase == "face":
            vertices = vertices[SMPLXSeg.head_ids + SMPLXSeg.neck_ids]
        max_v = vertices.max(0)[0]
        min_v = vertices.min(0)[0]
        scale = max_v[1] - min_v[1]
        center = (max_v + min_v) * 0.5
        # center = torch.mean(points, dim=0, keepdim=True)
        return center, scale

    @torch.no_grad()
    def export_mesh(self, save_dir):

        mesh, landmarks, keypoints, v_posed, joints, v_cano_dense = self.get_mesh(
            is_train=False, return_skel=True
        )

        obj_path = os.path.join(save_dir, "mesh.obj")
        mesh.write(obj_path)

        # save skeleton
        keypoint = keypoints.detach().cpu().numpy()
        keypoint_path = os.path.join(save_dir, "keypoint.npy")
        np.save(keypoint_path, keypoint)

        # save skeleton original order
        joints = joints.detach().cpu().numpy()
        joints_path = os.path.join(save_dir, "keypoint_ori.npy")
        np.save(joints_path, joints)

        # save vertex in cano order
        v_cano_dense = v_cano_dense.detach().cpu().numpy()
        v_cano_dense_path = os.path.join(save_dir, "v_cano_dense.npy")
        np.save(v_cano_dense_path, v_cano_dense)

        skel_vec = trimesh.Trimesh(vertices=keypoint[:18])
        skel_path = os.path.join(save_dir, "skel.obj")
        skel_vec.export(skel_path)

        # save normal mesh
        normal_mesh = trimesh.Trimesh(
            vertices=mesh.v.detach().cpu().numpy(),
            faces=mesh.f.detach().cpu().numpy(),
            vertex_colors=mesh.vn.detach().cpu().numpy() * 0.5 + 0.5,
        )

        normal_mesh_path = os.path.join(save_dir, "normal_mesh.ply")
        normal_mesh.export(normal_mesh_path)

    @torch.no_grad()
    def export_pose_mesh(
        self,
        save_dir,
    ):
        retarget_pose = self.motion_retarget(self.betas)

        pose_idx = np.arange(0, len(retarget_pose), 1)

        _, _, _, _, center, scale = self.get_mesh(
            is_train=False, return_scale=True, center=None, scale=None
        )

        for idx in pose_idx:
            mesh, landmarks, keypoints, v_posed = self.get_mesh(
                is_train=False,
                test_pose=retarget_pose[idx : idx + 1],
                center=center,
                scale=scale,
            )

            obj_path = os.path.join(save_dir, "mesh_%d.obj" % idx)
            mesh.write(obj_path)

    def forward(
        self,
        rays_o,
        rays_d,
        mvp,
        h,
        w,
        light_d=None,
        ambient_ratio=1.0,
        shading="albedo",
        is_train=True,
        test_pose=None,
        is_full_body=True,
    ):

        batch = rays_o.shape[0]

        bg_color = torch.ones(batch, h, w, 3).to(mvp.device)

        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = rays_o[0] + torch.randn(
                3, device=rays_o.device, dtype=torch.float
            )
            light_d = safe_normalize(light_d)

        retarget_pose = self.retarget_pose

        cano_mesh, _, cano_skeleton, _, center, scale = self.get_mesh(
            is_train=is_train, return_scale=True
        )

        pr_mesh, smplx_landmarks, smplx_skeleton, v_posed = self.get_mesh(
            is_train=is_train,
            pose_seq=retarget_pose,
            test_pose=test_pose,
            sample_cano=True,
            is_full_body=is_full_body,
            # use canonical center/scale
            center=center,
            scale=scale,
        )

        smplx_skeleton_copy = smplx_skeleton[:18]

        # skeleton map as condition
        smplx_skeleton = smplx_skeleton.reshape(-1, 3)
        skeleton = torch.bmm(
            F.pad(smplx_skeleton, pad=(0, 1), mode="constant", value=1.0).unsqueeze(0),
            torch.transpose(mvp, 1, 2),
        ).float()  # [B, N, 4]
        skeleton = skeleton[..., :2] / skeleton[..., 2:3]

        skeleton = skeleton * 0.5 + 0.5

        rgb, normal, alpha, rast, _ = self.renderer(
            pr_mesh,
            mvp,
            h,
            w,
            light_d,
            ambient_ratio,
            shading,
            self.opt.ssaa,
            mlp_texture=self.mlp_texture,
            is_train=is_train,
            return_rast=True,
        )
        rgb = rgb * alpha + (1 - alpha) * bg_color
        normal = normal * alpha + (1 - alpha) * bg_color

        ##### draw occlusion-aware skeleton
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)

        skeleton = 512 * (
            skeleton[
                :,
                :18,
            ]
            .detach()
            .cpu()
            .numpy()
        )
        # reference: https://github.com/facebookresearch/pytorch3d/issues/126
        pix_to_face = rast[..., -1:].clone().detach()
        pix_to_face[pix_to_face >= pr_mesh.f.shape[0]] = pr_mesh.f.shape[0] - 1
        packed_faces = pr_mesh.f

        visible_faces = torch.unique(pix_to_face).long()

        visible_verts_idx = torch.unique(packed_faces[visible_faces]).long()

        visible_verts = pr_mesh.v[visible_verts_idx]
        visible_verts_idx = visible_verts_idx.detach().cpu().numpy().tolist()

        for cur_joint in range(18):
            if cur_joint in self.face_indices:
                topk = 20
            else:  # body joints
                topk = 50

            x = smplx_skeleton_copy[cur_joint].detach()

            closet_in_verts = torch.sum((pr_mesh.v - x) ** 2, dim=1)

            idx = (
                closet_in_verts.topk(topk, largest=False)
                .indices.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            if not any([x in set(visible_verts_idx) for x in set(idx)]):
                skeleton[:, cur_joint] = None

        skeleton_condition = draw_bodypose(
            canvas,
            (skeleton),
        )

        if is_train:
            canvas = normal.detach().cpu().numpy()[0] * 255
            normal_skeleton = (
                draw_bodypose(
                    canvas,
                    (skeleton),
                )
                / 255
            )
            normal_skeleton = torch.from_numpy(normal_skeleton).unsqueeze(0).cuda()

        else:
            normal_skeleton = normal

        skeleton_condition = (
            torch.from_numpy(skeleton_condition)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(mvp.device)
        ) / 255.0

        # smplx landmarks
        smplx_landmarks = torch.bmm(
            F.pad(smplx_landmarks, pad=(0, 1), mode="constant", value=1.0).unsqueeze(0),
            torch.transpose(mvp, 1, 2),
        ).float()  # [B, N, 4]
        smplx_landmarks = smplx_landmarks[..., :2] / smplx_landmarks[..., 2:3]
        smplx_landmarks = smplx_landmarks * 0.5 + 0.5

        return {
            "image": rgb,
            "alpha": alpha,
            "normal": normal,
            "smplx_landmarks": smplx_landmarks,
            "skeleton_condition": skeleton_condition.detach(),
            "normal_skeleton": normal_skeleton,
            "mesh": pr_mesh,
            "cano_mesh": cano_mesh,
            "cano_skeleton": cano_skeleton,
        }

    def get_mesh_sequenes(self, is_train, pose_seq=None):
        assert (not self.opt.lock_geo) is True

        mesh_sequences = []
        vertex_sequences = []
        landmark_sequences = []
        align_face_sequences = []
        keypoints_sequences = []

        seq_length = 8
        sample_rate = 1

        sequence_index = random.randint(0, pose_seq.shape[0] - seq_length * sample_rate)

        pose_index = torch.arange(
            sequence_index, sequence_index + seq_length * sample_rate, sample_rate
        )

        # get the scale/center of canonical pose
        output_temp = self.body_model(
            betas=self.betas,
            body_pose=self.body_pose,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            return_verts=True,
        )
        v_cano_temp = output_temp.v_posed[0]

        # re-mesh
        v_cano_dense_temp = subdivide_inorder(
            v_cano_temp, self.smplx_faces[self.remesh_mask], self.uniques[0]
        )

        for unique, faces in zip(self.uniques[1:], self.faces_list[:-1]):
            v_cano_dense_temp = subdivide_inorder(v_cano_dense_temp, faces, unique)

        # add offset before warp
        if not self.opt.lock_geo:
            if self.v_offsets.shape[1] == 1:
                vn_temp = compute_normal(v_cano_dense_temp, self.faces_list[-1])[0]
                v_cano_dense_temp += self.get_vertex_offset(is_train) * vn_temp
            else:
                v_cano_dense_temp += self.get_vertex_offset(is_train)
        # LBS
        v_posed_dense_temp = warp_points(
            v_cano_dense_temp,
            self.dense_lbs_weights,
            output_temp.joints_transform[:, :55],
        ).squeeze(0)

        _, center, scale = normalize_vert(v_posed_dense_temp, return_cs=True)
        ######## end ######

        for i in range(pose_index.shape[0]):
            output = self.body_model(
                betas=self.betas,
                body_pose=(pose_seq[i : i + 1, 1:]),
                jaw_pose=self.jaw_pose,
                expression=self.expression,
                return_verts=True,
            )
            v_cano = output.v_posed[0]
            landmarks = output.joints[0, -68:, :]

            # re-mesh
            v_cano_dense = subdivide_inorder(
                v_cano, self.smplx_faces[self.remesh_mask], self.uniques[0]
            )

            for unique, faces in zip(self.uniques[1:], self.faces_list[:-1]):
                v_cano_dense = subdivide_inorder(v_cano_dense, faces, unique)

            # add offset before warp
            if not self.opt.lock_geo:
                if self.v_offsets.shape[1] == 1:
                    vn = compute_normal(v_cano_dense, self.faces_list[-1])[0]
                    v_cano_dense += self.get_vertex_offset(is_train) * vn
                else:
                    v_cano_dense += self.get_vertex_offset(is_train)
            # LBS
            v_posed_dense = warp_points(
                v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55]
            ).squeeze(0)

            v_posed_dense = (v_posed_dense - center) * scale

            # normalize v_posed, joints, and landmarks
            v_posed = (output.v_posed[0] - center) * scale

            # normalize joints
            joints = (output.joints[0] - center) * scale
            landmarks = (landmarks - center) * scale

            mesh = Mesh(
                v_posed_dense, self.faces_list[-1].int(), vt=self.vt, ft=self.ft
            )
            mesh.auto_normal()

            if not self.opt.lock_tex and not self.opt.tex_mlp:
                mesh.set_albedo(self.raw_albedo)

            mesh_sequences.append(mesh)
            vertex_sequences.append(v_posed_dense)
            landmark_sequences.append(landmarks)
            align_face_sequences.append(
                scale * (output.joints[0, 55, :][None, :] - center)
            )

            keypoints = joints[
                ...,
                smpl_to_openpose(
                    "smplx", openpose_format="coco18", use_face_contour=True
                ),
                :,
            ]

            keypoints_sequences.append(keypoints)

        return (
            mesh_sequences,
            vertex_sequences,
            landmark_sequences,
            align_face_sequences,
            keypoints_sequences,
        )

    def forward_seq(
        self,
        rays_o,
        rays_d,
        mvp,
        h,
        w,
        light_d=None,
        ambient_ratio=1.0,
        shading="albedo",
        is_train=True,
    ):

        batch = rays_o.shape[0]

        bg_color = torch.ones(batch, h, w, 3).to(mvp.device)

        if light_d is None:
            light_d = rays_o[0] + torch.randn(
                3, device=rays_o.device, dtype=torch.float
            )
            light_d = safe_normalize(light_d)

        # render
        pr_mesh, smplx_landmarks, cano_skeleton, v_posed = self.get_mesh(
            is_train=is_train
        )

        retarget_pose = self.retarget_pose

        (
            mesh_sequences,
            vertex_sequences,
            smplx_landmark_sequences,
            align_face_sequences,
            keypoints_sequences,
        ) = self.get_mesh_sequenes(is_train=is_train, pose_seq=retarget_pose)

        smplx_skeleton_sequences = torch.stack(keypoints_sequences, dim=0)

        rgb, normal, alpha, rast, _ = self.renderer.forward_sequences(
            pr_mesh,
            mvp,
            h,
            w,
            light_d,
            ambient_ratio,
            shading,
            self.opt.ssaa,
            mlp_texture=self.mlp_texture,
            is_train=is_train,
            vertex_sequences=vertex_sequences,
            return_rast=True,
        )
        rgb = rgb * alpha + (1 - alpha) * bg_color
        normal = normal * alpha + (1 - alpha) * bg_color

        skeleton_condition_sequences = []

        dense_face = pr_mesh.f.long()

        mask = torch.isin(
            torch.arange(25193), torch.from_numpy(self.smplx_face_vertix_idx)
        ).cuda()

        mask = mask[dense_face].all(axis=1).cuda()

        pix_to_face = rast[..., -1:].clone()
        face_render_mask = torch.isin(
            pix_to_face,
            torch.where(mask)[0],
        )  # b h w 1

        for seq_idx, (cur_mesh, smplx_skeleton) in enumerate(
            zip(mesh_sequences, keypoints_sequences)
        ):

            smplx_skeleton_copy = smplx_skeleton[:18]

            # skeleton map as condition
            smplx_skeleton = smplx_skeleton.reshape(-1, 3)
            skeleton = torch.bmm(
                F.pad(smplx_skeleton, pad=(0, 1), mode="constant", value=1.0).unsqueeze(
                    0
                ),
                torch.transpose(mvp, 1, 2),
            ).float()  # [B, N, 4]
            skeleton = skeleton[..., :2] / skeleton[..., 2:3]

            skeleton = skeleton * 0.5 + 0.5

            canvas = np.zeros((512, 512, 3), dtype=np.uint8)

            skeleton = 512 * (
                skeleton[
                    :,
                    :18,
                ]
                .detach()
                .cpu()
                .numpy()
            )
            # reference: https://github.com/facebookresearch/pytorch3d/issues/126
            pix_to_face = rast[..., -1:].clone().detach()
            pix_to_face[pix_to_face >= cur_mesh.f.shape[0]] = cur_mesh.f.shape[0] - 1
            packed_faces = cur_mesh.f

            visible_faces = torch.unique(pix_to_face).long()
            visible_verts_idx = torch.unique(packed_faces[visible_faces]).long()

            visible_verts_idx = visible_verts_idx.detach().cpu().numpy().tolist()

            for cur_joint in range(18):
                if cur_joint in self.face_indices:
                    topk = 20
                else:  # body joints
                    topk = 50

                x = smplx_skeleton_copy[cur_joint].detach()

                closet_in_verts = torch.sum((cur_mesh.v - x) ** 2, dim=1)

                idx = (
                    closet_in_verts.topk(topk, largest=False)
                    .indices.detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                if not any([x in set(visible_verts_idx) for x in set(idx)]):
                    skeleton[:, cur_joint] = None

            skeleton_condition = draw_bodypose(
                canvas,
                (skeleton),
            )

            skeleton_condition = (
                torch.from_numpy(skeleton_condition)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(mvp.device)
            ) / 255.0
            skeleton_condition_sequences.append(skeleton_condition)

        skeleton_condition_sequences = torch.concat(skeleton_condition_sequences, dim=0)

        return {
            "image": rgb,
            "alpha": alpha,
            "normal": normal,
            "skeleton_condition": skeleton_condition_sequences.detach(),
            "mesh": pr_mesh,
            "cano_skeleton": cano_skeleton,
            "face_render_mask": face_render_mask,
        }
