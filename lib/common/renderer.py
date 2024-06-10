import random

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from . import utils
from lib.common.obj import compute_normal


class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.glctx = dr.RasterizeCudaContext()
        except:
            self.glctx = dr.RasterizeGLContext()

    def forward(
        self,
        mesh,
        mvp,
        h=512,
        w=512,
        light_d=None,
        ambient_ratio=1.0,
        shading="albedo",
        spp=1,
        mlp_texture=None,
        is_train=False,
        return_rast=False,
    ):
        B = mvp.shape[0]
        v_clip = torch.bmm(
            F.pad(mesh.v, pad=(0, 1), mode="constant", value=1.0).unsqueeze(0).expand(B, -1, -1),
            torch.transpose(mvp, 1, 2),
        ).float()  # [B, N, 4]

        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)

        ################################################################################
        # Interpolate attributes
        ################################################################################

        # Interpolate world space position
        alpha, _ = dr.interpolate(torch.ones_like(v_clip[..., :1]), rast, mesh.f)  # [B, H, W, 1]
        depth = rast[..., [2]]  # [B, H, W]

        if is_train:
            vn, _ = compute_normal(v_clip[0, :, :3], mesh.f)
            normal, _ = dr.interpolate(vn[None, ...].float(), rast, mesh.f)
        else:
            normal, _ = dr.interpolate(mesh.vn[None, ...].float(), rast, mesh.f)

        # Texture coordinate
        if not shading == "normal":
            albedo = self.get_2d_texture(mesh, rast, rast_db)

        if shading == "normal":
            color = (normal + 1) / 2.0
        elif shading == "albedo":
            color = albedo
        else:  # lambertian
            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d.view(-1, 1)).float().clamp(min=0)
            color = albedo * lambertian.repeat(1, 1, 1, 3)

        normal = (normal + 1) / 2.0

        normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]

        # inverse super-sampling
        if spp > 1:
            color = utils.scale_img_nhwc(color, (h, w))
            alpha = utils.scale_img_nhwc(alpha, (h, w))
            normal = utils.scale_img_nhwc(normal, (h, w))

        if return_rast:
            return color, normal, alpha, rast, rast_db

        return color, normal, alpha

    @staticmethod
    def get_2d_texture(mesh, rast, rast_db):
        texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs="all")

        albedo = dr.texture(
            mesh.albedo.unsqueeze(0),
            texc,
            uv_da=texc_db,
            filter_mode="linear-mipmap-linear",
        )  # [B, H, W, 3]
        albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background
        return albedo

    def forward_sequences(
        self,
        mesh,
        mvp,
        h=512,
        w=512,
        light_d=None,
        ambient_ratio=1.0,
        shading="albedo",
        spp=1,
        mlp_texture=None,
        is_train=False,
        vertex_sequences=None,
        # align_face_sequences=None,
        return_rast=False,
    ):
        B = mvp.shape[0]
        v_clip = torch.bmm(
            F.pad(mesh.v, pad=(0, 1), mode="constant", value=1.0).unsqueeze(0).expand(B, -1, -1),
            torch.transpose(mvp, 1, 2),
        ).float()  # [B, N, 4]

        vertex_sequences = torch.stack(vertex_sequences, dim=0)

        B_v = vertex_sequences.shape[0]
        mvp = mvp.expand(B_v, -1, -1)

        v_clip = torch.bmm(
            F.pad(vertex_sequences, pad=(0, 1), mode="constant", value=1.0),
            torch.transpose(mvp, 1, 2),
        ).float()  # [B, N, 4]

        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)

        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)

        ################################################################################
        # Interpolate attributes
        ################################################################################

        # Interpolate world space position
        alpha, _ = dr.interpolate(torch.ones_like(v_clip[..., :1]), rast, mesh.f)  # [B, H, W, 1]

        if is_train:
            vn_sequences = []
            for i in range(B_v):
                vn, _ = compute_normal(v_clip[i, :, :3], mesh.f)
                vn_sequences.append(vn)
            vn_sequences = torch.stack(vn_sequences, dim=0)

            normal, _ = dr.interpolate(vn_sequences.float(), rast, mesh.f)

        else:
            assert False

        # Texture coordinate
        if not shading == "normal":
            albedo = self.get_2d_texture(mesh, rast, rast_db)

        if shading == "normal":
            color = (normal + 1) / 2.0
        elif shading == "albedo":
            color = albedo
        else:  # lambertian
            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d.view(-1, 1)).float().clamp(min=0)
            color = albedo * lambertian.repeat(1, 1, 1, 3)

        normal = (normal + 1) / 2.0

        normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]

        # inverse super-sampling
        if spp > 1:
            color = utils.scale_img_nhwc(color, (h, w))
            alpha = utils.scale_img_nhwc(alpha, (h, w))
            normal = utils.scale_img_nhwc(normal, (h, w))

        if return_rast:
            return color, normal, alpha, rast, rast_db

        return color, normal, alpha
