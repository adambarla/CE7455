from datetime import datetime
import wandb
from tqdm import tqdm
import hydra
import numpy as np
import random
from omegaconf import OmegaConf, DictConfig
from torchtext import data
import torch
from torchtext import datasets


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


def epoch_train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    total_correct = 0
    total_instances = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        # Assuming your model's forward method automatically handles padding, then no need to pack sequence here
        predictions = model(text, text_lengths)

        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Compute the number of correct predictions
        _, predicted_classes = predictions.max(dim=1)
        correct_predictions = (
            predicted_classes == batch.label
        ).float()  # Convert to float for summation
        total_correct += correct_predictions.sum().item()
        total_instances += batch.label.size(0)

    return epoch_loss / len(iterator), total_correct / total_instances


def epoch_evaluate(model, iterator, criterion):
    epoch_loss = 0
    total_correct = 0
    total_instances = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            # Assuming your model's forward method automatically handles padding, then no need to pack sequence here
            predictions = model(text, text_lengths)

            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()

            # Compute the number of correct predictions
            _, predicted_classes = predictions.max(dim=1)
            correct_predictions = (
                predicted_classes == batch.label
            ).float()  # Convert to float for summation
            total_correct += correct_predictions.sum().item()
            total_instances += batch.label.size(0)

    epoch_acc = total_correct / total_instances
    return epoch_loss / len(iterator), epoch_acc


def train(model, optimizer, criterion, n_epochs, train_iterator, valid_iterator):
    with tqdm(total=n_epochs, desc="Training Progress") as pbar:
        for epoch in range(n_epochs):
            train_loss, train_acc = epoch_train(
                model, train_iterator, optimizer, criterion
            )
            valid_loss, valid_acc = epoch_evaluate(model, valid_iterator, criterion)
            pbar.set_description(
                f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%"
                f" |  Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |"
            )
            pbar.update(1)
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                }
            )


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


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    init_run(cfg)
    set_deterministic(cfg.seed)
    device = get_device(cfg)
    train_data, valid_data, test_data, TEXT, LABEL = load_data(cfg.seed)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=cfg.batch_size,
        sort_within_batch=True,
        device=device,
    )
    model = hydra.utils.instantiate(
        cfg.model, vocab_size=len(TEXT.vocab), output_dim=len(LABEL.vocab)
    )
    print(model)
    model.to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, optimizer, criterion, cfg.epochs, train_iterator, valid_iterator)


if __name__ == "__main__":
    main()
