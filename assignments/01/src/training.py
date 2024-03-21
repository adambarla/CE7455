import wandb
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from torchtext import data
import torch
from utils import get_device, init_run, load_data, set_deterministic


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
