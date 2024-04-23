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
    max_grad_norm,
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
                V = out_lang.n_words
                optimizer.zero_grad()
                out_tok, out_prob = model(inputs, targets)
                loss = criterion(out_prob.view(-1, V), targets[1:].view(-1))
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
    n = min(n, len(loader))
    inputs, targets = next(iter(loader))
    sen_in = input_lang.decode(inputs.transpose(0, 1))
    sen_tgt = output_lang.decode(targets.transpose(0, 1))
    out_tok, out_prob = model(inputs, targets, use_teacher_forcing=False)
    sen_out = output_lang.decode(out_tok.transpose(0, 1))
    for i in range(n):
        print(">", sen_in[i])
        print("=", sen_tgt[i])
        print("<", sen_out[i])
        print("")


def test(
    model,
    output_lang,
    loader,
    criterion,
    name,
    n=10,
):
    model.eval()
    rouge = torchmetrics.text.rouge.ROUGEScore()  # todo: refactor
    hypothesis = []
    references = []
    loss_sum = 0
    with torch.no_grad():
        with tqdm(total=len(loader), desc=f"Testing {name} partion") as bar:
            for i, (inputs, targets) in enumerate(loader):
                V = output_lang.n_words
                out_tok, out_prob = model(inputs, targets, use_teacher_forcing=False)
                loss = criterion(out_prob.view(-1, V), targets[1:].view(-1))
                loss_sum += loss.item()
                out_sentences = output_lang.decode(
                    out_tok.transpose(0, 1)
                )  # L x B -> B x L
                tgt_sentences = output_lang.decode(targets.transpose(0, 1))
                hypothesis.extend(out_sentences)
                references.extend([[s] for s in tgt_sentences])
                bar.update(1)
                bar.set_postfix(loss=loss_sum / (i + 1))
            bar.close()
    for i in range(min(n, len(hypothesis))):
        print("<", hypothesis[i])
        print(">", references[i][0])
        print("")
    rs = rouge(hypothesis, references)
    metrics = {k: v.item() for k, v in rs.items()}
    metrics["loss"] = loss_sum / len(loader)
    longest_name = max(len(k) for k in metrics.keys())
    s = f"Results for {name} partition"
    print("-" * len(s))
    print(s)
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
        cfg,
        device,
    )
    encoder = hydra.utils.instantiate(
        cfg.encoder, input_size=in_lang.n_words, device=device
    ).to(device)
    decoder = hydra.utils.instantiate(
        cfg.decoder,
        output_size=out_lang.n_words,
        device=device,
        hidden_size=encoder.hidden_size * (2 if cfg.encoder.bidirectional else 1),
        base={
            "hidden_size": encoder.hidden_size * (2 if cfg.encoder.bidirectional else 1),
            "input_size": encoder.hidden_size * (2 if cfg.encoder.bidirectional else 1)
        },
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
        max_grad_norm=cfg.max_grad_norm,
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
