import torch
import torch.nn as nn

import torch.nn.functional as F

from einops import rearrange

import torchvision.transforms as T

from lib.guidance.t2v_model.pipeline_conditionvideo import ConditionVideoPipeline

from lib.guidance.t2v_model.unet import UNet3DConditionModel
from lib.guidance.t2v_model.controlnet import ControlNet3DModel


from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)


def prepare_models():
    control_models = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float32,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=control_models,
        safety_checker=None,
        torch_dtype=torch.float32,
    ).to(torch.device("cuda"))

    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    vae = pipe.vae

    unet = UNet3DConditionModel.from_pretrained_2d(pipe.unet.cpu()).cuda()

    controlnet = ControlNet3DModel.from_pretrained_2d(pipe.controlnet.cpu()).cuda()

    controlnet.cond_weight = 1.0

    controlnet.requires_grad_(False)
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()

    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()

    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "unet": unet,
        "controlnet": controlnet,
        "scheduler": scheduler,
    }


class T2VSDS(ConditionVideoPipeline):

    def __init__(
        self,
        t_range=[0.02, 0.98],
        weighting_strategy="fantasia3d",
    ):

        pipeline_kwargs = prepare_models()

        super().__init__(**pipeline_kwargs)

        generator = torch.Generator(device=torch.device("cuda"))
        # generator.manual_seed(seed)

        self.generator = generator

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.weighting_strategy = weighting_strategy

        self.alphas = self.scheduler.alphas_cumprod.to(torch.device("cuda"))  # for convenience

    def encode_images(self, images):

        images = 2 * images - 1  # [B, 3, H, W]
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def sequential_noise(self, latent_seq):
        b, c, f, h, w = latent_seq.shape

        shape_image = (b, c, h, w)
        latents = torch.randn(shape_image, generator=self.generator, device="cuda")

        latents = torch.stack([latents] * f, dim=2)  # b c f h w

        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def train_step(
        self,
        text_embeddings,
        pred_rgb,
        cond_inputs=None,
        guidance_scale=100,
        rgb_as_latents=False,
        ddim_inv_latent=None,
        height=512,
        width=512,
        fix_image_noise=True,
        # eta=0.0,
        face_masks=None,
    ):

        whole_conds = cond_inputs

        cond_idx = 0
        cond = whole_conds[cond_idx].unsqueeze(dim=0)  # (b f c h w)

        batch_size = pred_rgb.shape[0]
        num_videos_per_prompt = 1
        num_channels_latents = self.unet.in_channels
        video_length = cond_inputs.shape[1]

        device = torch.device("cuda")

        b, f, c, h, w = pred_rgb.shape

        pred_rgb = pred_rgb.view(b * f, c, h, w)

        # latents = []
        if rgb_as_latents:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False)
            latents = latents * 2 - 1
        else:
            # ramdomly select several frames do not require grad
            vnum = 4

            mask_vae = torch.randperm(pred_rgb.shape[0]) < vnum

            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)

            with torch.no_grad():
                latents_wo_grad = self.encode_images(pred_rgb_512[~mask_vae])

            latents_w_grad = self.encode_images(pred_rgb_512[mask_vae])

            latents = torch.zeros(
                pred_rgb_512.shape[0],
                *latents_w_grad.shape[1:],
                device=latents_w_grad.device,
                dtype=latents_w_grad.dtype,
            )

            latents[~mask_vae] = latents_wo_grad
            latents[mask_vae] = latents_w_grad

        latents = latents.view(b, f, -1, 64, 64).permute(0, 2, 1, 3, 4)

        latents = torch.mean(latents, keepdim=True, dim=0)

        # prepare condition latent
        cond_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            self.generator,
            None,
            fix_image_noise=fix_image_noise,
        )

        # Denoising loop

        down_block_res_samples_processed = None
        mid_block_res_sample = None
        # cond = cond.to(device)
        cond = rearrange(cond, "b t c h w -> b c t h w")
        cond = torch.cat([cond] * 2)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (latents.shape[0],),
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():

            # 1. Add Noise
            # noise = torch.randn_like(latents)
            noise = self.sequential_noise(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # 2. Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents_noisy] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            cond_latent_model_input = torch.cat([cond_latents] * 2)
            cond_latent_model_input = self.scheduler.scale_model_input(cond_latent_model_input, t)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                cond_latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=cond,
                return_dict=False,
            )

            down_block_res_samples_processed = ()
            for down_block_res_sample in down_block_res_samples:
                down_block_res_sample = down_block_res_sample * self.controlnet.cond_weight
                down_block_res_samples_processed += (down_block_res_sample,)
            mid_block_res_sample = mid_block_res_sample * self.controlnet.cond_weight

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples_processed,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

        # perform guidance (high scale from paper!)
        # do clf-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        # squeeze batch axis
        noise_pred = noise_pred.squeeze(0)
        noise = noise.squeeze(0)

        if self.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        grad = w * (noise_pred - noise)

        grad = torch.nan_to_num(grad)

        if face_masks is not None:
            bb, tt, cc, hh, ww = face_masks.shape
            face_masks = face_masks.reshape(bb * tt, cc, hh, ww)
            resize = T.Resize(
                (latents.shape[-2], latents.shape[-1]),
                interpolation=T.InterpolationMode.NEAREST,
            )

            face_masks = resize(face_masks)
            face_masks = face_masks.reshape(bb, tt, cc, 64, 64)
            face_masks = rearrange(face_masks, "b t c h w -> b c t h w")

            body_masks = 1 - face_masks

            loss = (
                0.5
                * F.mse_loss(
                    body_masks * latents,
                    body_masks * (latents - grad).detach(),
                    reduction="sum",
                )
                / latents.shape[0]
                / latents.shape[2]  # div b, t
            )

        else:
            loss = (
                0.5
                * F.mse_loss(latents, (latents - grad).detach(), reduction="sum")
                / latents.shape[0]
                / latents.shape[2]
            )

        return loss
