import wandb
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from torchtext import data
import torch
from utils import count_correct, set_deterministic, init_run, load_data, get_device


def epoch_train(
    model, iterator, optimizer, criterion, regularizer=None, grad_clip_threshold=None
):
    epoch_loss = 0
    total_correct = 0
    total_instances = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths)

        loss = criterion(predictions, batch.label)
        if regularizer is not None:
            loss += regularizer(model)
        loss.backward()
        if grad_clip_threshold is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)
        optimizer.step()
        epoch_loss += loss.item()
        total_correct += count_correct(batch.label, predictions)
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
            predictions = model(text, text_lengths)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
            total_correct += count_correct(batch.label, predictions)
            total_instances += batch.label.size(0)

    epoch_acc = total_correct / total_instances
    return epoch_loss / len(iterator), epoch_acc


def train(
    model,
    optimizer,
    criterion,
    n_epochs,
    train_iterator,
    valid_iterator,
    test_iterator,
    regularizer=None,
    patience=torch.inf,
    grad_clip_threshold=None,
):
    max_acc = -1
    epochs_since_improvement = 0
    with tqdm(total=n_epochs, desc="Training Progress") as pbar:
        for epoch in range(n_epochs):
            train_loss, train_acc = epoch_train(
                model,
                train_iterator,
                optimizer,
                criterion,
                regularizer,
                grad_clip_threshold,
            )
            valid_loss, valid_acc = epoch_evaluate(model, valid_iterator, criterion)
            pbar.set_description(
                f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%"
                f" |  Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% "
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
            if valid_acc > max_acc:
                max_acc = valid_acc
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                break
    if epochs_since_improvement >= patience:
        print(
            f"Early stopping triggered in epoch {epoch + 1} because val/acc hasn't improved for {epochs_since_improvement} epochs."
        )
    test_loss, test_acc = epoch_evaluate(model, test_iterator, criterion)
    print(f" Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%")
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})


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
    base = hydra.utils.instantiate(cfg.base)
    print(base)
    if base.bidirectional:
        cfg.classifier.input_dim = base.hidden_size * 2
    classifier = hydra.utils.instantiate(
        cfg.classifier,
        output_dim=len(LABEL.vocab),
    )
    print(classifier)
    model = hydra.utils.instantiate(
        cfg.model,
        vocab_size=len(TEXT.vocab),
        output_dim=len(LABEL.vocab),
        TEXT=TEXT,
        base=base,
        classifier=classifier,
    )
    print(model)
    model.to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()
    regularizer = (
        hydra.utils.instantiate(cfg.regularizer) if cfg.regularizer.use else None
    )
    patience = cfg.patience if cfg.patience is not None else torch.inf
    train(
        model,
        optimizer,
        criterion,
        cfg.epochs,
        train_iterator,
        valid_iterator,
        test_iterator,
        regularizer,
        patience,
        cfg.grad_clip_threshold,
    )


if __name__ == "__main__":
    main()
