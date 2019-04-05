import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

hidden_size = 7
num_layers = 1
bidirectional = False
batch_size = 4
time_step = 5
embed_size = 6

lstm = nn.LSTM(input_size=embed_size,
               hidden_size=hidden_size,
               num_layers=num_layers,
               bias=True,
               batch_first=True,
               dropout=0,
               bidirectional=bidirectional)

input = torch.ones(batch_size, time_step, embed_size)
input_lens = torch.zeros(batch_size).fill_(time_step).long()
packed_input = pack_padded_sequence(input, input_lens, batch_first=True)
output, (hidden, cell) = lstm(packed_input)
print(hidden.size())