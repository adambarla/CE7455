from datetime import datetime
import numpy as np
import random
import torch
import wandb
from omegaconf import OmegaConf, DictConfig


def get_device(cfg: DictConfig) -> torch.device:
    if cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return device


def init_run(cfg):
    if cfg.name is None:
        # m = cfg.model._target_.split(".")[-1]
        # o = cfg.optimizer._target_.split(".")[-1]
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        # cfg.name = f"{m}_{o}_{t}"
        cfg.name = f"{t}"
    wandb.init(
        name=cfg.name,
        group=cfg.group,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
