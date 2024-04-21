import time

import numpy as np
import torchmetrics
from torch import nn
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from utils import (
    load_data,
    set_deterministic,
    init_run,
    get_device,
    time_since,
)


def train_iters(
    model,
    epochs,
    loader,
    print_every,
    criterion,
    optimizer,
):
    model.train()
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    i = 1
    n_iters = len(loader) * epochs
    for epoch in range(epochs):
        print("Epoch: %d/%d" % (epoch, epochs))
        for inputs, targets in tqdm(loader):
            inputs = inputs[0]
            targets = targets[0]
            optimizer.zero_grad()
            outputs = model(inputs, targets)
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            optimizer.step()
            print_loss_total += loss.item()
            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print(
                    "%s (%d %d%%) %.4f"
                    % (
                        time_since(start, i / n_iters),
                        i,
                        i / n_iters * 100,
                        print_loss_avg,
                    )
                )
            i += 1


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
        outputs = model.predict(inputs)
        print("<", output_lang.decode(outputs))
        print("")


def test(
    model,
    output_lang,
    loader,
):
    rouge = torchmetrics.text.rouge.ROUGEScore()  # todo: refactor
    all_inputs = []
    gt = []
    predict = []
    metric_score = {
        "rouge1_fmeasure": [],
        "rouge1_precision": [],
        "rouge1_recall": [],
        "rouge2_fmeasure": [],
        "rouge2_precision": [],
        "rouge2_recall": [],
    }
    for i, (inputs, targets) in tqdm(enumerate(loader)):
        inputs = inputs[0]
        targets = targets[0]
        outputs = model.predict(inputs)
        output_sentence = output_lang.decode(outputs)
        all_inputs.append(inputs)
        gt.append(targets)
        predict.append(output_sentence)
        try:
            rs = rouge(output_sentence, targets)
        except:
            continue
        metric_score["rouge1_fmeasure"].append(rs["rouge1_fmeasure"])
        metric_score["rouge1_precision"].append(rs["rouge1_precision"])
        metric_score["rouge1_recall"].append(rs["rouge1_recall"])
        metric_score["rouge2_fmeasure"].append(rs["rouge2_fmeasure"])
        metric_score["rouge2_precision"].append(rs["rouge2_precision"])
        metric_score["rouge2_recall"].append(rs["rouge2_recall"])
    metric_score["rouge1_fmeasure"] = np.array(metric_score["rouge1_fmeasure"]).mean()
    metric_score["rouge1_precision"] = np.array(metric_score["rouge1_precision"]).mean()
    metric_score["rouge1_recall"] = np.array(metric_score["rouge1_recall"]).mean()
    metric_score["rouge2_fmeasure"] = np.array(metric_score["rouge2_fmeasure"]).mean()
    metric_score["rouge2_precision"] = np.array(metric_score["rouge2_precision"]).mean()
    metric_score["rouge2_recall"] = np.array(metric_score["rouge2_recall"]).mean()
    print("=== Evaluation score - Rouge score ===")
    print("Rouge1 fmeasure:\t", metric_score["rouge1_fmeasure"])
    print("Rouge1 precision:\t", metric_score["rouge1_precision"])
    print("Rouge1 recall:  \t", metric_score["rouge1_recall"])
    print("Rouge2 fmeasure:\t", metric_score["rouge2_fmeasure"])
    print("Rouge2 precision:\t", metric_score["rouge2_precision"])
    print("Rouge2 recall:  \t", metric_score["rouge2_recall"])
    print("=====================================")
    return all_inputs, gt, predict, metric_score


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    init_run(cfg)
    set_deterministic(cfg.seed)
    device = get_device(cfg)
    train_loader, test_loader, input_lang, output_lang = load_data(
        cfg.seed,
        cfg.l1,
        cfg.l2,
        cfg.test_size,
        cfg.sos_token,
        cfg.eos_token,
        cfg.max_length,
        device,
        cfg.batch_size,
    )
    encoder = hydra.utils.instantiate(
        cfg.encoder, input_size=input_lang.n_words, device=device
    ).to(device)
    decoder = hydra.utils.instantiate(
        cfg.decoder,
        output_size=output_lang.n_words,
        device=device,
    ).to(device)
    model = hydra.utils.instantiate(
        cfg.model,
        encoder=encoder,
        decoder=decoder,
        device=device,
    )
    criterion = nn.NLLLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    train_iters(
        model,
        epochs=cfg.epochs,
        loader=train_loader,
        print_every=cfg.print_every,
        criterion=criterion,
        optimizer=optimizer,
    )
    evaluate_randomly(
        model,
        input_lang,
        output_lang,
        test_loader,
        n=10,
    )
    test(model, output_lang, train_loader)
    test(
        model,
        output_lang,
        test_loader,
    )


if __name__ == "__main__":
    main()
