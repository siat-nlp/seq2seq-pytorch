from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.encoder import Encoder
from src.module.utils.constants import PAD_INDEX
from src.module.utils.clone import clone

class RecurrentEncoder(Encoder):

    def __init__(self, embedding, layer, num_layers, dropout):
        super(RecurrentEncoder, self).__init__()
        self.embedding = embedding
        self.dropout = dropout
        self.layers = clone(layer, num_layers)
        hidden_size = layer.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = dropout

    def forward(self, src):
        src_embedding = self.embedding(src)
        src_embedding = F.dropout(src_embedding, p=self.dropout, training=self.training)
        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        src_lens, sort_index = src_lens.sort(descending=True)
        src_embedding = src_embedding.index_select(index=sort_index, dim=0)
        packed_src = pack_padded_sequence(src_embedding, src_lens, batch_first=True)
        final_states = []
        for layer in self.layers:
            packed_src, final_state = layer(packed_src)
            final_states.append(final_state)
        src, _ = pad_packed_sequence(packed_src, batch_first=True)
        src = self.layer_norm(src)
        return src, src_mask, final_states

class RecurrentEncoderLayer(nn.Module):

    def __init__(self, rnn, feed_forward, dropout):
        super(RecurrentEncoderLayer, self).__init__()
        hidden_size = rnn.hidden_size
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.rnn = rnn
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src, src_lens = pad_packed_sequence(src, batch_first=True)
        src = self.layer_norm1(src)
        residual = src
        src = pack_padded_sequence(src, src_lens, batch_first=True)
        src, final_state = self.rnn(src)
        src, _ = pad_packed_sequence(src, batch_first=True)
        src = src + residual
        src = self.dropout1(src)
        src = self.layer_norm2(src)
        src = src + self.feed_forward(src)
        src = self.dropout2(src)
        src = pack_padded_sequence(src, src_lens, batch_first=True)
        if isinstance(final_state, tuple):
            final_state = (self.feed_forward(final_state[0]), self.feed_forward(final_state[1]))
        else:
            final_state = self.feed_forward(final_state)
        return src, final_state