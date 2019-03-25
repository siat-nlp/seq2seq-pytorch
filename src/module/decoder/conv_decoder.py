from torch import nn
import torch.nn.functional as F
import math
from src.module.decoder.decoder import Decoder
from src.module.utils.constants import PAD_INDEX
from src.module.utils.clone import clone

class ConvDecoder(Decoder):

    def __init__(self, embedding, positional_embedding, layer, num_layers, dropout):
        super(ConvDecoder, self).__init__()
        self.embedding = embedding
        self.positional_embedding = positional_embedding
        embed_size = embedding.embedding_dim
        hidden_size = layer.hidden_size
        self.input_projection = nn.Linear(embed_size, hidden_size)
        self.layers = clone(layer, num_layers)
        self.output_projection = nn.Linear(hidden_size, embed_size)
        self.dropout = dropout

    def forward(self, src, trg):
        return self.step(src, trg)

    def greedy_decode(self, src, max_len):
        pass

    def beam_decode(self, src, max_len, beam_size):
        pass

    def step(self, src, trg_slice):
        trg_slice_embedding = self.embedding(trg_slice) + self.positional_embedding(trg_slice)
        trg_slice_embedding = F.dropout(trg_slice_embedding, p=self.dropout, training=self.training)
        trg_slice_mask = (trg_slice != PAD_INDEX)
        trg_slice = trg_slice_embedding.transpose(0, 1)
        trg_slice = self.input_projection(trg_slice)
        for layer in self.layers:
            trg_slice = layer(src, (trg_slice, trg_slice_mask))
        trg_slice = self.output_projection(trg_slice)
        trg_slice = trg_slice.transpose(0, 1)
        logits = trg_slice.matmul(self.embedding.weight.t())
        return logits

class ConvDecoderLayer(nn.Module):

    def __init__(self, conv, attention, feed_forward, dropout):
        super(ConvDecoderLayer, self).__init__()
        hidden_size = conv.hidden_size
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.conv = conv
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.attention = attention
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, src_mask, trg, trg_mask):
        pass