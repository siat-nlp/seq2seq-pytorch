import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.encoder.encoder import Encoder
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
        src = src_embedding
        for layer in self.layers:
            src = layer(src)
        src = self.layer_norm(src)
        return src

class RecurrentEncoderLayer(nn.Module):

    def __init__(self):
        super(RecurrentEncoderLayer, self).__init__()

    def forward(self, x):
        pass