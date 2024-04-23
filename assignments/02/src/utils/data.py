import os
import re
import unicodedata
import zipfile
from io import BytesIO
import requests
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def download_and_extract(lang1, lang2):
    url = f"http://www.manythings.org/anki/{lang1}-{lang2}.zip"
    extract_to = f"data/{lang1}-{lang2}/"
    if os.path.exists(extract_to + f"{lang1}.txt"):
        print("Data already present.")
        return
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise Exception(
            f"Failed to download data, status code: {r.status_code}, url: {url}"
        )
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall(extract_to)
    print("Downloaded and extracted data.")


class LangDataset(Dataset):
    def __init__(self, pairs, input_lang, output_lang, device):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inputs, targets = self.pairs[idx]
        input_tensor = self.tensor_from_sentence(self.input_lang, inputs)
        target_tensor = self.tensor_from_sentence(self.output_lang, targets)
        return input_tensor, target_tensor

    def tensor_from_sentence(self, lang, sentence):
        indexes = lang.encode(sentence)
        indexes = [lang.sos_token] + indexes + [lang.eos_token]
        tensor = torch.tensor(indexes, dtype=torch.long, device=self.device)
        return tensor


def load_data(
    cfg,
    device,
):
    download_and_extract(cfg.l1, cfg.l2)
    in_lang, out_lang, pairs = prepare_data(cfg, reverse=cfg.reverse)
    x = [i[0] for i in pairs]
    y = [i[1] for i in pairs]
    assert cfg.val_size + cfg.test_size < 1
    fraction = cfg.test_size + cfg.val_size
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        x, y, test_size=fraction, random_state=cfg.seed
    )
    fraction = cfg.test_size / (cfg.test_size + cfg.val_size)
    x_va, x_te, y_va, y_te = train_test_split(
        x_tmp, y_tmp, test_size=fraction, random_state=cfg.seed
    )
    tr_pairs = list(zip(x_tr, y_tr))
    va_pairs = list(zip(x_va, y_va))
    te_pairs = list(zip(x_te, y_te))
    tr_dataset = LangDataset(tr_pairs, in_lang, out_lang, device)
    va_dataset = LangDataset(va_pairs, in_lang, out_lang, device)
    te_dataset = LangDataset(te_pairs, in_lang, out_lang, device)
    collator = Collator(cfg.pad_token)

    tr_loader = DataLoader(
        tr_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator
    )
    va_loader = DataLoader(
        va_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator
    )
    te_loader = DataLoader(
        te_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator
    )
    return tr_loader, va_loader, te_loader, in_lang, out_lang


class Collator(object):
    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, batch):
        inputs, targets = zip(*batch)
        inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=False, padding_value=self.pad_token
        )
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=False, padding_value=self.pad_token
        )
        return inputs, targets


class Lang:
    def __init__(self, name, sos_token, eos_token, pad_token, unk_token):
        self.name = name
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2index = {
            "<s>": sos_token,
            "</s>": eos_token,
            "<pad>": pad_token,
            "<unk>": unk_token,
        }
        self.index2word = {
            sos_token: "<s>",
            eos_token: "</s>",
            pad_token: "<pad>",
            unk_token: "<unk>",
        }
        self.special_tokens = {sos_token, eos_token, pad_token, unk_token}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def encode(self, sentence):
        return [
            self.word2index.get(word, self.unk_token) for word in sentence.split(" ")
        ]

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(x, list) and (not x or isinstance(x[0], int)):
            x = [x]
        return [
            " ".join(
                self.index2word.get(t, self.index2word[self.unk_token])
                for t in seq
                if t not in self.special_tokens
            )
            for seq in x
        ]


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(cfg, reverse=False):
    lines = (
        open(f"data/{cfg.l1}-{cfg.l2}/{cfg.l1}.txt", encoding="utf-8")
        .read()
        .strip()
        .split("\n")
    )
    pairs = [[normalize_string(s) for s in l.split("\t")[:2]] for l in lines]
    l1 = cfg.l1 if not reverse else cfg.l2
    l2 = cfg.l2 if not reverse else cfg.l1
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
    input_lang = Lang(l1, cfg.sos_token, cfg.eos_token, cfg.pad_token, cfg.unk_token)
    output_lang = Lang(l2, cfg.sos_token, cfg.eos_token, cfg.pad_token, cfg.unk_token)
    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am",
    "i m",
    "he is",
    "he s",
    "she is",
    "she s",
    "you are",
    "you re",
    "we are",
    "we re",
    "they are",
    "they re",
)


def filter_pair(p, max_length):
    return (
        len(p[0].split(" ")) < max_length
        and len(p[1].split(" ")) < max_length
        and p[1].startswith(eng_prefixes)
    )


def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length=max_length)]


def prepare_data(cfg, reverse):
    input_lang, output_lang, pairs = read_langs(cfg, reverse=reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs, cfg.max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
