from transformers import logging
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

from diffusers.utils import load_image
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Union

import sys


class ControlNet(nn.Module):
    def __init__(
        self,
        device,
        latent_mode=True,
        concept_name=None,
        guidance_scale=100.0,
        fp16=True,
    ):
        super().__init__()

        self.device = device
        self.latent_mode = latent_mode
        self.num_train_timesteps = 1000
        self.guidance_scale = guidance_scale

        assert concept_name is None

        control_models = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose",
            torch_dtype=torch.float16 if fp16 else torch.float32,
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=control_models,
            safety_checker=None,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).to(self.device)

        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.controlnet = self.pipe.controlnet

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """
        Args:
            prompt: str

        Returns:
            text_embeddings: torch.Tensor
        """
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings

    def produce_latents(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=None,
        latents=None,
    ):
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents

    def encode_images(self, images):
        images = 2 * images - 1  # [B, 3, H, W]
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    def decode_latents(self, latents, to_uint8=False, to_numpy=False):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        with torch.no_grad():
            latents = 1 / self.vae.config.scaling_factor * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        if to_uint8:
            imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
        if to_numpy:
            imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
        return imgs

    def calc_condition(self, image_path):
        results = []
        image = load_image(image_path)
        for processor in self.cond_processors:
            # process image with each condition processor
            cond = processor(image)
            # store the result
            results.append(cond)
        return results

    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                image = [np.array(i.resize((width, height), resample=Image.Resampling.LANCZOS))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    def prompt_to_img(
        self,
        prompts,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=None,
        latents=None,
    ):
        # Prompts -> text embeds: [unconditioned embedding, text embedding]
        if isinstance(prompts, torch.Tensor):
            text_embeds = prompts
        else:
            if isinstance(prompts, str):
                prompts = [prompts]
            text_embeds = self.get_text_embeds(prompts)  # [2, 77, 768]

        # Text embeds -> img latents: [1, 4, 64, 64]
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        latents = self.produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Img latents -> images
        images = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()  # [1, 512, 512, 3]
        images = (images * 255).round().astype("uint8")

        return images

    def batched_prompt_to_img(self, prompts, batch_size=4, **kwargs):
        if isinstance(prompts, torch.Tensor):
            assert prompts.size(0) % 2 == 0  # [z_uncond, z_cond]
            num_samples = prompts.size(0) // 2
            uncond_embeds_list = torch.split(prompts[:num_samples], batch_size)
            cond_embeds_list = torch.split(prompts[num_samples:], batch_size)
        else:
            raise NotImplementedError
        images_list = []
        for uncond_embeds, cond_embeds in zip(uncond_embeds_list, cond_embeds_list):
            text_embeds = torch.cat((uncond_embeds, cond_embeds))
            images = self.prompt_to_img(text_embeds, **kwargs)
            images_list.append(images)
        return np.concatenate(images_list)

    def batch_prompt_control_to_img(
        self,
        prompts,
        cond_inputs,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=None,
        controlnet_conditioning_scale=1.0,
        dtype=torch.float,
    ):
        assert isinstance(prompts, str) or isinstance(prompts, List(str))
        batch_size = 1 if isinstance(prompts, str) else len(prompts)

        # prepare cond_inputs
        assert isinstance(self.controlnet, ControlNetModel)

        cond_inputs = self.prepare_image(
            cond_inputs,
            height,
            width,
            batch_size=batch_size,
            num_images_per_prompt=1,
            device=self.device,
            dtype=dtype,
        )

        # inference
        image = self.pipe(
            prompts,
            num_inference_steps=num_inference_steps,
            image=cond_inputs,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="np.array",
        ).images  # [1, 512, 512, 3]

        # Img to Numpy
        return (image * 255).round().astype("uint8")


class T2ISDS(ControlNet):

    def __init__(
        self,
        device,
        latent_mode=True,
        concept_name=None,
        guidance_scale=100,
        fp16=True,
        t_range=[0.02, 0.98],
        weighting_strategy="fantasia3d",
    ):
        super().__init__(device, latent_mode, concept_name, guidance_scale, fp16)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.weighting_strategy = weighting_strategy

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """
        Args:
            prompt: str

        Returns:
            text_embeddings: torch.Tensor
        """
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings

    def train_step(
        self,
        text_embeddings,
        pred_rgb,
        cond_inputs=None,
        guidance_scale=100,
        rgb_as_latents=False,
        step_ratio=None,
        return_noise=False,
    ):
        if rgb_as_latents:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False)
            latents = latents * 2 - 1
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            latents = self.encode_images(pred_rgb_512)

        latents = torch.mean(latents, keepdim=True, dim=0)

        batch_size = latents.size(0)

        cond_inputs = self.prepare_image(
            cond_inputs,
            512,
            512,
            batch_size=batch_size,
            num_images_per_prompt=1,
            device=self.device,
            dtype=pred_rgb.dtype,
        )

        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (latents.shape[0],),
            dtype=torch.long,
            device=self.device,
        )

        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # add controlnet condition
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=cond_inputs,
                conditioning_scale=1.0,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

        # perform guidance (high scale from paper!)
        # do clf-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        if self.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        #
        grad = w * (noise_pred - noise)

        grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(latents, (latents - grad).detach(), reduction="sum") / latents.shape[0]

        if return_noise:
            return loss, noise_pred
        else:
            return loss
