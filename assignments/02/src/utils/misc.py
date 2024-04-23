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
    e = cfg.encoder.base._target_.split(".")[-1] if cfg.encoder.get("base") else cfg.encoder._target_.split(".")[-1]
    d = cfg.decoder.base._target_.split(".")[-1] if cfg.decoder.get("base") else cfg.decoder._target_.split(".")[-1]
    # m = cfg.model._target_.split(".")[-1]
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.name is None:
        cfg.name = f"{e}_{d}_{t}"

    if cfg.group is None:
        cfg.group = f"{e}_{d}"
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
