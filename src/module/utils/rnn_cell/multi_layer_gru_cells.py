import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerGRUCells(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bias=True):
        super(MultiLayerGRUCells, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bias = bias
        self._dropout = dropout
        self._gru_cells = nn.ModuleList([nn.GRUCell(self._input_size, self._hidden_size, self._bias)])
        self._gru_cells.extend(nn.ModuleList(
            nn.GRUCell(self._hidden_size, self._hidden_size, self._bias)
            for _ in range(self._num_layers - 1)
        ))

    def forward(self, input, states):
        # input: Tensor (batch_size, input_size)
        # states: hidden
        # hidden: Tensor (num_layers, batch_size, hidden_size)
        hidden = states
        output_hidden = []
        for i, gru_cell in enumerate(self._gru_cells):
            h = gru_cell(input, hidden[i])
            output_hidden.append(h)
            input = F.dropout(h, p=self._dropout, training=self.training)
        output_hidden = torch.stack(output_hidden, dim=0)
        return output_hidden

    @property
    def input_size(self):
        return self._input_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def bias(self):
        return self._bias