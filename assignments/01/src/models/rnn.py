import torch
import gensim.downloader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from utils import is_sorted, get_embedding_matrix


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, **kwargs):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class PackedRNN(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0, **kwargs
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # dropout in nn.RNN is not applied oh hidden state only on outputs of all layers except the last layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        # assert is_sorted(text_lengths)
        embedded = self.embedding(text)
        packed_input = pack_padded_sequence(embedded, text_lengths)
        packed_output, hidden = self.rnn(packed_input)
        hidden = self.dropout(hidden)
        return self.fc(hidden.squeeze(0))


class GensimPackedRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        TEXT,
        dropout=0,
        **kwargs
    ):
        super().__init__()
        word_vectors = gensim.downloader.load("word2vec-google-news-300")
        embedding_matrix = get_embedding_matrix(TEXT, word_vectors)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        # dropout in nn.RNN is not applied oh hidden state only on outputs of all layers except the last layer
        self.rnn = nn.RNN(300, hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        # assert is_sorted(text_lengths)
        embedded = self.embedding(text)
        packed_input = pack_padded_sequence(embedded, text_lengths)
        packed_output, hidden = self.rnn(packed_input)
        hidden = self.dropout(hidden)
        return self.fc(hidden.squeeze(0))
