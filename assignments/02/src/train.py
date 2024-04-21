import random
import time

import numpy as np
import torchmetrics
from torch import nn
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
from utils import (
    load_data,
    set_deterministic,
    init_run,
    get_device,
    tensors_from_pair,
    time_since,
    tensor_from_sentence,
)


def train_iters(
    model,
    epochs,
    train_pairs,
    input_lang,
    output_lang,
    device,
    eos_token,
    print_every,
    criterion,
    optimizer,
):
    model.train()
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    i = 1
    n_iters = len(train_pairs) * epochs
    for epoch in range(epochs):
        print("Epoch: %d/%d" % (epoch, epochs))
        for training_pair in tqdm(train_pairs):
            optimizer.zero_grad()
            training_pair = tensors_from_pair(
                training_pair, input_lang, output_lang, device, eos_token
            )
            inputs = training_pair[0]
            targets = training_pair[1]
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


def evaluate(
    encoder,
    decoder,
    sentence,
    max_length,
    device,
    input_lang,
    output_lang,
    sos_token,
    eos_token,
):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence, device, eos_token)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[sos_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            top_v, top_i = decoder_output.data.topk(1)
            if top_i.item() == eos_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[top_i.item()])
            decoder_input = top_i.squeeze().detach()
        return decoded_words


def evaluate_randomly(
    encoder,
    decoder,
    input_lang,
    output_lang,
    pairs,
    device,
    max_length,
    sos_token,
    eos_token,
    n=10,
):
    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words = evaluate(
            encoder,
            decoder,
            pair[0],
            max_length,
            device,
            input_lang,
            output_lang,
            sos_token,
            eos_token,
        )
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def test(
    encoder,
    decoder,
    input_lang,
    output_lang,
    testing_pairs,
    max_length,
    device,
    sos_token,
    eos_token,
):
    rouge = torchmetrics.text.rouge.ROUGEScore()  # todo: refactor
    inputs = []
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
    for i in tqdm(range(len(testing_pairs))):
        pair = testing_pairs[i]
        output_words = evaluate(
            encoder,
            decoder,
            pair[0],
            max_length,
            device,
            input_lang,
            output_lang,
            sos_token,
            eos_token,
        )
        output_sentence = " ".join(output_words)
        inputs.append(pair[0])
        gt.append(pair[1])
        predict.append(output_sentence)
        try:
            rs = rouge(output_sentence, pair[1])
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
    return inputs, gt, predict, metric_score


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    init_run(cfg)
    set_deterministic(cfg.seed)
    device = get_device(cfg)
    train_pairs, test_pairs, input_lang, output_lang = load_data(
        cfg.seed,
        cfg.l1,
        cfg.l2,
        cfg.test_size,
        cfg.sos_token,
        cfg.eos_token,
        cfg.max_length,
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
        cfg.model, encoder=encoder, decoder=decoder, device=device
    )
    criterion = nn.NLLLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    train_iters(
        model,
        epochs=cfg.epochs,
        train_pairs=train_pairs,
        input_lang=input_lang,
        output_lang=output_lang,
        device=device,
        eos_token=cfg.eos_token,
        print_every=cfg.print_every,
        criterion=criterion,
        optimizer=optimizer,
    )
    evaluate_randomly(
        encoder,
        decoder,
        input_lang,
        output_lang,
        test_pairs,
        device,
        cfg.max_length,
        cfg.sos_token,
        cfg.eos_token,
        n=10,
    )
    test(
        encoder,
        decoder,
        input_lang,
        output_lang,
        test_pairs,
        cfg.max_length,
        device,
        cfg.sos_token,
        cfg.eos_token,
    )
    test(
        encoder,
        decoder,
        input_lang,
        output_lang,
        test_pairs,
        cfg.max_length,
        device,
        cfg.sos_token,
        cfg.eos_token,
    )


if __name__ == "__main__":
    main()
