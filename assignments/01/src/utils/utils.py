from datetime import datetime
from torchtext import data
import numpy as np
import random

import torch
import wandb
from torchtext import datasets
from omegaconf import OmegaConf, DictConfig


def is_sorted(sequence):
    return all(sequence[i] >= sequence[i + 1] for i in range(len(sequence) - 1))


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
        m = cfg.model._target_.split(".")[-1]
        o = cfg.optimizer._target_.split(".")[-1]
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.name = f"{m}_{o}_{t}"
    wandb.init(
        name=cfg.name,
        group=cfg.group,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )


def load_data(seed):
    # For tokenization
    TEXT = data.Field(
        tokenize="spacy", tokenizer_language="en_core_web_sm", include_lengths=True
    )

    # For multi-class classification labels
    LABEL = data.LabelField()

    # Load the TREC dataset
    train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)
    train_data, valid_data = train_data.split(
        split_ratio=0.8, random_state=random.seed(seed)
    )

    print(f"Number of training examples: {len(train_data)}")
    print(f"Number of validation examples: {len(valid_data)}")
    print(f"Number of testing examples: {len(test_data)}")

    TEXT.build_vocab(train_data, max_size=10000)
    LABEL.build_vocab(train_data)

    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    return train_data, valid_data, test_data, TEXT, LABEL


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_correct(labels, predictions):
    _, predicted_classes = predictions.max(dim=1)
    correct_predictions = (predicted_classes == labels).float()
    return correct_predictions.sum().item()


def get_embedding_matrix(TEXT, word_vectors):
    embedding_dim = 300
    num_embeddings = len(TEXT.vocab)

    embedding_matrix = torch.zeros(num_embeddings, embedding_dim)

    words_found = 0
    words_not_found = 0
    for i, word in enumerate(TEXT.vocab.itos):
        if word in word_vectors:
            embedding_vector = word_vectors[word]
            words_found += 1
        else:
            # If the word is not in pre-trained word vectors, initialize it with zeros.
            embedding_vector = np.zeros(embedding_dim)
            words_not_found += 1

        embedding_matrix[i] = torch.tensor(embedding_vector)

    print(f"Words found in pre-trained vectors: {words_found}")
    print(f"Words not found in pre-trained vectors: {words_not_found}")
    return embedding_matrix
