import torch
from torch import nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        max_length,
        sos_token,
        eos_token,
        pad_token,
        device,
        teacher_forcing_ratio=0.5,
    ):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_length = max_length
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    def forward(self, x, y, use_teacher_forcing=True):
        L, B = y.shape
        V = self.decoder.output_size
        e_out, e_hidden = self.encoder(x)
        d_hidden = e_hidden
        d_out_prob = torch.zeros(L - 1, B, V, device=self.device)
        d_out_tok = torch.full(
            (L, B), self.pad_token, device=self.device, dtype=torch.long
        )
        d_in = torch.full((1, B), self.sos_token, device=self.device, dtype=torch.long)
        for i in range(L - 1):
            d_out_prob[i], d_hidden = self.decoder(d_in, d_hidden)
            _, top_i = d_out_prob[i].topk(1, dim=1)
            top_i = top_i.view(1, B)
            d_out_tok[i] = d_in
            if use_teacher_forcing:
                msk = (
                    torch.rand((1, B), device=self.device) < self.teacher_forcing_ratio
                )
                d_in = torch.where(msk, y[i + 1].view(1, B), top_i)
            else:
                d_in = top_i
        return d_out_tok, d_out_prob


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, base, device):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.base = base
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        L, B = x.shape
        embedded = self.embedding(x).view(L, B, -1)
        output, hidden = self.base(embedded)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, base, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.base = base
        self.out = nn.Linear(hidden_size, output_size)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.out.weight)
        self.out.bias.data.fill_(0)

    def forward(self, x, hidden):
        assert hidden is not None, "Hidden state is required for decoder"
        L, B = x.shape
        # todo: Your code here
        embedded = self.embedding(x).view(L, B, -1)
        output = F.relu(embedded)
        output, hidden = self.base(output, hidden)  # output is L x B x H
        output = self.out(output[0])
        output = F.log_softmax(output, dim=1)
        if torch.isnan(output[0]).any():
            raise ValueError("NaN detected in output")
        return output, hidden
