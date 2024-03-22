import hydra
import torch
import gensim.downloader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from utils import is_sorted, get_embedding_matrix


class FC(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout, **kwargs):
        super().__init__()
        hidden_dims = [input_dim] + hidden_dims
        self.layers = nn.Sequential(
            *[
                self._block(hidden_dims[i], hidden_dims[i + 1], dropout)
                for i in range(len(hidden_dims) - 1)
            ],
            nn.Linear(hidden_dims[-1], output_dim),
        )

    def _block(self, input_dim, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


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
        **kwargs,
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


class AttentionGensimPackedRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        TEXT,
        base,
        classifier,
        dropout=0,
        n_heads=1,
        **kwargs,
    ):
        super().__init__()
        word_vectors = gensim.downloader.load("word2vec-google-news-300")
        embedding_matrix = get_embedding_matrix(TEXT, word_vectors)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        # dropout in nn.RNN is not applied oh hidden state only on outputs of all layers except the last layer
        # instantiate base RNN class
        self.base = base
        if base.bidirectional:
            hidden_dim = hidden_dim * 2
        # self.rnn = nn.RNN(300, hidden_dim, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads)
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.classifier = classifier

    def forward(self, text, text_lengths):
        # assert is_sorted(text_lengths)
        embedded = self.embedding(text)
        packed_input = pack_padded_sequence(embedded, text_lengths)
        packed_output, _ = self.base(packed_input)
        padded_output, _ = pad_packed_sequence(packed_output)
        attention_output, _ = self.attention(
            padded_output, padded_output, padded_output
        )
        sentence_embedding = torch.mean(attention_output, dim=0)
        sentence_embedding = self.dropout(sentence_embedding)
        return self.classifier(sentence_embedding.squeeze(0))
