from torch import nn
import torch.nn.functional as F
from src.module.layer.feed_forward import FeedForward

class RecurrentDecoderLayer(nn.Module):

    def __init__(self, rnn_type, hidden_size, dropout=0.1,
                 input_size='default', feed_forward_size='default'):
        super(RecurrentDecoderLayer, self).__init__()
        input_size = hidden_size if input_size == 'default' else input_size
        feed_forward_size = 4 * hidden_size if feed_forward_size == 'default' else feed_forward_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        if rnn_type == 'LSTM':
            self.rnn_cell = nn.LSTMCell(
                input_size=input_size,
                hidden_size=hidden_size
            )
        elif rnn_type == 'GRU':
            self.rnn_cell = nn.GRUCell(
                input_size=input_size,
                hidden_size=hidden_size
            )
        else:
            raise ValueError('%s is not supported.' % rnn_type)
        self.feed_forward = FeedForward(
            input_size=hidden_size,
            feed_forward_size=feed_forward_size,
            output_size=hidden_size
        )
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = dropout

    def forward(self, input, initial_state):
        input = self.layer_norm1(input)
        final_state = self.rnn_cell(input, initial_state)
        if self.rnn_type == 'LSTM':
            hidden = final_state[0]
        else:
            hidden = final_state
        # hidden = hidden + input
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.layer_norm2(hidden)
        hidden = self.feed_forward(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden, final_state