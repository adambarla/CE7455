from datetime import datetime
import numpy as np
import random
import torch
import wandb
from omegaconf import OmegaConf, DictConfig
import time
import math


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


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


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence, device, eos_token):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(eos_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair, input_lang, output_lang, device, eos_token):
    input_tensor = tensor_from_sentence(input_lang, pair[0], device, eos_token)
    target_tensor = tensor_from_sentence(output_lang, pair[1], device, eos_token)
    return input_tensor, target_tensor


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
