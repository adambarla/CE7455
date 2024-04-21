import torch
import torchmetrics
import wandb
from torch import nn
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from utils import (
    load_data,
    set_deterministic,
    init_run,
    get_device,
)


def train_iters(
    model,
    epochs,
    tr_loader,
    va_loader,
    criterion,
    optimizer,
    out_lang,
    early_stopping,
):
    loss_sum = 0
    i = 1
    for epoch in range(epochs):
        model.train()
        s = f"Epoch: {epoch:>{len(str(epochs))}}/{epochs}"
        print("=" * len(s))
        print(s)
        with tqdm(total=len(tr_loader), desc="Training") as bar:
            for inputs, targets in tr_loader:
                inputs = inputs[0]
                targets = targets[0]
                optimizer.zero_grad()
                outputs = model(inputs, targets)
                loss = criterion(outputs, targets.squeeze())
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                wandb.log({"train_loss": loss_sum / i})
                bar.set_postfix(loss=loss_sum / i)
                bar.update(1)
                i += 1
            bar.close()
        metrics = test(model, out_lang, va_loader, criterion, "valid")
        if early_stopping.should_stop(metrics):
            print(f"Early stopping triggered in epoch {epoch + 1}")
            break


def evaluate_randomly(
    model,
    input_lang,
    output_lang,
    loader,
    n=10,
):
    for i, (inputs, targets) in enumerate(loader):
        if i >= n:
            break
        inputs = inputs[0]
        targets = targets[0]
        print(">", input_lang.decode(inputs))
        print("=", output_lang.decode(targets))
        outputs, _ = model.predict(inputs)
        print("<", output_lang.decode(outputs))
        print("")


def test(
    model,
    output_lang,
    loader,
    criterion,
    name,
):
    model.eval()
    rouge = torchmetrics.text.rouge.ROUGEScore()  # todo: refactor
    hypothesis = []
    references = []
    loss_sum = 0
    with torch.no_grad():
        with tqdm(total=len(loader), desc=f"Testing {name} partion") as bar:
            for i, (inputs, targets) in enumerate(loader):
                inputs = inputs[0]
                targets = targets[0]
                outputs, probs = model.predict(inputs)
                loss = criterion(probs[: len(targets)], targets.squeeze())
                loss_sum += loss.item()
                output_sentence = output_lang.decode(outputs)
                target_sentence = output_lang.decode(targets)
                hypothesis.append(output_sentence)
                references.append([target_sentence])
                bar.update(1)
                bar.set_postfix(loss=loss_sum / (i + 1))
            bar.close()
    rs = rouge(hypothesis, references)
    metrics = {k: v.item() for k, v in rs.items()}
    metrics["loss"] = loss_sum / len(loader)
    longest_name = max(len(k) for k in metrics.keys())
    for k, v in metrics.items():
        print(f"{k:>{longest_name}}: {v:.4g}")
    wandb.log({f"{name}_{k}": v for k, v in metrics.items()})
    return metrics


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    init_run(cfg)
    set_deterministic(cfg.seed)
    device = get_device(cfg)
    tr_loader, va_loader, te_loader, in_lang, out_lang = load_data(
        cfg.seed,
        cfg.l1,
        cfg.l2,
        cfg.test_size,
        cfg.val_size,
        cfg.sos_token,
        cfg.eos_token,
        cfg.max_length,
        device,
        cfg.batch_size,
    )
    encoder = hydra.utils.instantiate(
        cfg.encoder, input_size=in_lang.n_words, device=device
    ).to(device)
    decoder = hydra.utils.instantiate(
        cfg.decoder,
        output_size=out_lang.n_words,
        device=device,
    ).to(device)
    model = hydra.utils.instantiate(
        cfg.model,
        encoder=encoder,
        decoder=decoder,
        device=device,
    ).to(device)
    criterion = nn.NLLLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    print(f"Model: {model}\nCriterion: {criterion}\nOptimizer: {optimizer}")
    early_stopping = hydra.utils.instantiate(cfg.early_stopping)
    train_iters(
        model,
        epochs=cfg.epochs,
        tr_loader=tr_loader,
        va_loader=va_loader,
        criterion=criterion,
        optimizer=optimizer,
        out_lang=out_lang,
        early_stopping=early_stopping,
    )
    evaluate_randomly(
        model,
        in_lang,
        out_lang,
        te_loader,
        n=10,
    )
    test(model, out_lang, tr_loader, criterion, "train")
    test(
        model,
        out_lang,
        te_loader,
        criterion,
        "test",
    )


if __name__ == "__main__":
    main()
