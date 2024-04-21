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

    def forward(self, x, y):
        input_length = x.size(0)
        target_length = y.size(0)
        # batch_size = y.size(1)
        vocab_size = self.decoder.output_size
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=self.device
        )
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[i], encoder_hidden)
            encoder_outputs[i] = encoder_output[0, 0]
        decoder_input = torch.tensor([[self.sos_token]], device=self.device)
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(target_length, vocab_size, device=self.device)
        for i in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[i] = decoder_output
            if torch.rand(1).item() < self.teacher_forcing_ratio:
                decoder_input = y[i]
            else:
                top_v, top_i = decoder_output.topk(1)
                decoder_input = top_i.squeeze().detach()
                if decoder_input.item() == self.eos_token:
                    break
        return decoder_outputs

    def predict(self, x):
        input_length = x.size(0)
        # vocab_size = self.decoder.out.out_features
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=self.device
        )
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[i], encoder_hidden)
            encoder_outputs[i] = encoder_output[0, 0]
        decoder_input = torch.tensor([[self.sos_token]], device=self.device)
        decoder_hidden = encoder_hidden
        outputs = []
        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            top_v, top_i = decoder_output.topk(1)
            if top_i.item() == self.eos_token:
                outputs.append(self.eos_token)
                break
            outputs.append(top_i.item())
            decoder_input = top_i.squeeze().detach()
        return outputs


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, base, device):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.base = base

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)
        output = embedded
        output, hidden = self.base(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # todo: Your code here
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)  # L x H_out, 1 x H_out
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
