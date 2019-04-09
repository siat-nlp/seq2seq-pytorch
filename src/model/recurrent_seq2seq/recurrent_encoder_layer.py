import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.layer.feed_forward import FeedForward

class RecurrentEncoderLayer(nn.Module):

    def __init__(self, rnn_type, hidden_size, bidirectional=True, dropout=0.1,
                 input_size='default', feed_forward_size='default'):
        super(RecurrentEncoderLayer, self).__init__()
        input_size = hidden_size if input_size == 'default' else input_size
        feed_forward_size = 4 * hidden_size if feed_forward_size == 'default' else feed_forward_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            raise ValueError('%s is not supported.' % rnn_type)
        self.feed_forward = FeedForward(
            input_size=2 * hidden_size if bidirectional else hidden_size,
            feed_forward_size=feed_forward_size,
            output_size=hidden_size
        )
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = dropout

    def forward(self, packed_src):
        src, src_lens = pad_packed_sequence(packed_src, batch_first=True)
        src = self.layer_norm1(src)
        residual = src
        packed_src = pack_padded_sequence(src, src_lens, batch_first=True)
        packed_src, final_state = self.rnn(packed_src)
        src, _ = pad_packed_sequence(packed_src, batch_first=True)
        src = src + residual
        src = F.dropout(src, p=self.dropout, training=self.training)
        src = self.layer_norm2(src)
        src = src + self.feed_forward(src)
        src = F.dropout(src, p=self.dropout, training=self.training)
        packed_src = pack_padded_sequence(src, src_lens, batch_first=True)
        if self.rnn_type == 'LSTM':
            final_state = (
                torch.cat(final_state[0].split(split_size=1, dim=0), dim=2),
                torch.cat(final_state[1].split(split_size=1, dim=0), dim=2)
            )
        else:  # GRU
            final_state = torch.cat(final_state.split(split_size=1, dim=0), dim=2)
        return packed_src, final_state

class ExtendedRecurrentEncoderLayer(nn.Module):

    def __init__(self, rnn, feed_forward, dropout=0.1):
        super(ExtendedRecurrentEncoderLayer, self).__init__()
        hidden_size = rnn.hidden_size
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.rnn = rnn
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout = dropout

    def forward(self, src):
        src, src_lens = pad_packed_sequence(src, batch_first=True)
        src = self.layer_norm1(src)
        residual = src
        src = pack_padded_sequence(src, src_lens, batch_first=True)
        src, final_state = self.rnn(src)
        src, _ = pad_packed_sequence(src, batch_first=True)
        src = src + residual
        src = F.dropout(src, p=self.dropout, training=self.training)
        src = self.layer_norm2(src)
        src = src + self.feed_forward(src)
        src = F.dropout(src, p=self.dropout, training=self.training)
        src = pack_padded_sequence(src, src_lens, batch_first=True)
        if isinstance(final_state, tuple):  # LSTM
            final_state = (self.feed_forward(final_state[0]), self.feed_forward(final_state[1]))
        else:   # GRU
            final_state = self.feed_forward(final_state)
        return src, final_state