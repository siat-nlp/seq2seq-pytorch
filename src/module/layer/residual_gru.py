import torch
from torch import nn
import torch.nn.functional as F
from src.module.utils.clone import clone
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ResidualGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=True):
        super(ResidualGRU, self).__init__()
        self.lstm_layers = clone(nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional
        ), num_layers)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def forward(self, input, initial_states):
        state_layers = self.num_layers * (2 if self.bidirectional else 1)
        assert state_layers == initial_states.size(0)
        initial_states = torch.chunk(initial_states, chunks=self.num_layers, dim=0)
        input, input_lens = pad_packed_sequence(input, self.batch_first)
        input = self.input_projection(input)
        final_states = []
        for i, lstm_layer in self.lstm_layers:
            residual = input
            input = pack_padded_sequence(input, input_lens, self.batch_first)
            initial_state = initial_states[i]
            output, final_state = lstm_layer(input, initial_state)
            final_states.append(final_state)
            output, _ = pad_packed_sequence(output, self.batch_first)
            output = residual + output
            if i < self.num_layers - 1:
                output = F.dropout(output, p=self.dropout, training=self.training)
            input = output
        output = pack_padded_sequence(output, input_lens, self.batch_first)
        final_states = torch.cat(final_states, dim=0)
        return output, final_states