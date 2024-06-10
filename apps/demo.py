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
    # parser.add_argument('--mesh', type=str, required=True, help="mesh template, must be obj format")
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
            text=object_text,
            negative=cfg.negative,
            dir_text=cfg.data.dir_text,
            opt=cfg.training,
            model=model,
            guidance_t2i=None,
            device=device,
            fp16=cfg.fp16,
            exp_name=args.description,
        )

        test_loader = build_dataloader("test")

        print("test ing")
        trainer.test(test_loader)
        trainer.test_sequence_step(test_loader)

        if cfg.save_mesh:
            trainer.save_mesh()
            # trainer.save_pose_mesh()
