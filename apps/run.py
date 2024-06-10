import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.provider import ViewDataset
from lib.trainer import *
from lib.dlmesh import DLMesh
from lib.common.utils import load_config

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--text", default=None, help="text prompt")
    parser.add_argument("--negative", default="", help="negative text prompt")
    parser.add_argument("--description", default="", help="exp save folder")
    parser.add_argument("--t2m_model", default="mdiffuse", type=str, help="t2m models")

    args = parser.parse_args()

    cfg = load_config(args.config, "configs/default.yaml")

    cfg.merge_from_list(
        [
            "text",
            args.text,
            "negative",
            args.negative,
        ]
    )
    cfg.freeze()

    seed_everything(cfg.seed)

    def build_dataloader(phase):
        """
        Args:
            phase: str one of ['train', 'test' 'val']
        Returns:
        """
        size = 4 if phase == "val" else 100
        dataset = ViewDataset(cfg.data, device=device, type=phase, size=size)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def configure_t2i_guidance():
        from lib.guidance.controlnet_t2i import T2ISDS

        diffusion_model = T2ISDS(
            device=torch.device("cuda"),
            latent_mode=False,
            # fp16=False,
            fp16=cfg.fp16,
        )

        for p in diffusion_model.parameters():
            p.requires_grad = False

        return diffusion_model

    def configure_t2v_guidance():

        from lib.guidance.controlnet_t2v import T2VSDS

        diffusion_model = T2VSDS()
        diffusion_model.enable_vae_slicing()

        return diffusion_model

    def configure_optimizer():
        opt = cfg.training
        if opt.optim == "adan":
            from lib.common.optimizer import Adan

            optimizer = lambda model: Adan(
                model.get_params(5 * opt.lr),
                eps=1e-8,
                weight_decay=2e-5,
                max_grad_norm=5.0,
                foreach=False,
            )
        else:  # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(5 * opt.lr), betas=(0.9, 0.99), eps=1e-15)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.1 ** min(x / opt.iters, 1))
        return scheduler, optimizer

    text_split = cfg.text.split(", he/she")
    if len(text_split) == 1:
        object_text = text_split[0]
        motion_text = "A person is dancing."
    else:
        object_text, motion_text = text_split[0], text_split[1]
        motion_text = "A person" + motion_text
    model = DLMesh(cfg.model, text_prompt=motion_text, t2m_model=args.t2m_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.test:
        trainer = Trainer(
            cfg.name,
            # text=cfg.text,
            text=object_text,
            negative=cfg.negative,
            dir_text=cfg.data.dir_text,
            opt=cfg.training,
            model=model,
            guidance=None,
            device=device,
            fp16=cfg.fp16,
            exp_name=args.description,
        )

        test_loader = build_dataloader("test")

        trainer.test(test_loader)
        trainer.test_sequence_step(test_loader, is_init_pose=True)
        trainer.test_sequence_step(test_loader)

        if cfg.save_mesh:
            trainer.save_mesh()

        print("test is done")

    else:
        train_loader = build_dataloader("train")

        scheduler, optimizer = configure_optimizer()

        guidance_t2i = configure_t2i_guidance()
        guidance_t2v = configure_t2v_guidance()

        trainer = Trainer(
            cfg.name,
            # text=cfg.text,
            text=object_text,
            negative=cfg.negative,
            dir_text=cfg.data.dir_text,
            opt=cfg.training,
            model=model,
            guidance_t2i=guidance_t2i,
            device=device,
            optimizer=optimizer,
            fp16=cfg.fp16,
            lr_scheduler=scheduler,
            scheduler_update_every_step=True,
            exp_name=args.description,
            guidance_t2v=guidance_t2v,
        )
        if os.path.exists(cfg.data.image):
            trainer.default_view_data = train_loader.dataset.get_default_view_data()

        valid_loader = build_dataloader("val")
        max_epoch = np.ceil(cfg.training.iters / (len(train_loader) * train_loader.batch_size)).astype(np.int32)

        trainer.train(train_loader, valid_loader, max_epoch)

        # test
        test_loader = build_dataloader("test")
        trainer.test(test_loader)
        trainer.test_sequence_step(test_loader)
        trainer.save_mesh()
