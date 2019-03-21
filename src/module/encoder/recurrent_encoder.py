import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.encoder.encoder import Encoder

class RecurrentEncoder(Encoder):

    def __init__(self, rnn_type, embedding, hidden_size, num_layers, bidirectional, dropout):
        super(RecurrentEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.embedding = embedding
        state_layers = num_layers * (2 if bidirectional else 1)
        embed_size = embedding.embedding_dim
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
            self.init_states = nn.ParameterList([
                nn.Parameter(
                    torch.FloatTensor(state_layers, hidden_size)
                ),
                nn.Parameter(
                    torch.FloatTensor(state_layers, hidden_size)
                )
            ])
            init.xavier_uniform_(self.init_states[0])
            init.xavier_uniform_(self.init_states[1])
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
            self.init_states = nn.Parameter(
                torch.FloatTensor(state_layers, hidden_size)
            )
            init.xavier_uniform_(self.init_states)
        else:
            raise ValueError('No Supporting.')

    def forward(self, src):
        src_lens = (src != 0).long().sum(dim=1, keepdim=False)
        batch_size = src.size(0)
        init_states = self._get_init_states(batch_size)
        src_embedding = self.embedding(src)
        packed_src, sort_index = self._pack_padded_sequence(src_embedding, src_lens)
        packed_output, final_states = self.rnn(packed_src, init_states)
        output, final_states = self._pad_packed_sequence(packed_output, final_states, sort_index)
        return output, final_states

    def _get_init_states(self, batch_size):
        if self.rnn_type == 'LSTM':  # LSTM
            state_layers, hidden_size = self.init_states[0].size()
            size = (state_layers, batch_size, hidden_size)
            init_states = (
                self.init_states[0].unsqueeze(1).expand(*size).contiguous(),
                self.init_states[1].unsqueeze(1).expand(*size).contiguous()
            )
        else:  # GRU
            state_layers, hidden_size = self.init_states.size()
            size = (state_layers, batch_size, hidden_size)
            init_states = self.init_states.unsqueeze(1).expand(*size).contiguous()
        return init_states

    def _pack_padded_sequence(self, src_embedding, src_lens):
        src_lens, sort_index = src_lens.sort(descending=True)
        src_embedding = src_embedding.index_select(index=sort_index, dim=0)
        packed_src = pack_padded_sequence(src_embedding, src_lens, batch_first=True)
        return packed_src, sort_index

    def _pad_packed_sequence(self, packed_output, final_states, sort_index):
        desort_index = sort_index.argsort(descending=True)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = output.index_select(index=desort_index, dim=0)
        if self.rnn_type == 'LSTM':  # LSTM
            final_states = (
                final_states[0].index_select(index=desort_index, dim=1),
                final_states[1].index_select(index=desort_index, dim=1)
            )
        else:  # GRU
            final_states = final_states.index_select(index=desort_index, dim=1)
        return output, final_states