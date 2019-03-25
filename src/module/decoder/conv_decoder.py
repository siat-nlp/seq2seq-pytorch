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
        vocab_size = embedding.num_embeddings
        hidden_size = layer.hidden_size
        self.input_projection = nn.Linear(embed_size, hidden_size)
        self.layers = clone(layer, num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, embed_size)
        self.dropout = dropout
        self.generator = nn.Linear(embed_size, vocab_size)

    def forward(self, src, trg):
        return self.step(src, trg)

    def greedy_decode(self, src, max_len):
        pass

    def beam_decode(self, src, max_len, beam_size):
        pass

    def step(self, src, trg):
        trg_embedding = self.embedding(trg) + self.positional_embedding(trg)
        trg_embedding = F.dropout(trg_embedding, p=self.dropout, training=self.training)
        trg_mask = (trg != PAD_INDEX).transpose(0, 1)
        trg = trg_embedding.transpose(0, 1)
        trg = trg.masked_fill(trg_mask.unsqueeze(-1)==0, 0)
        trg = self.input_projection(trg)
        for layer in self.layers:
            trg = layer(src, (trg, trg_mask))
        trg = self.layer_norm(trg)
        trg = self.output_projection(trg)
        trg = trg.transpose(0, 1)
        logit = self.generator(trg)
        return logit

class ConvDecoderLayer(nn.Module):

    def __init__(self, hidden_size, conv, attention, feed_forward, dropout):
        super(ConvDecoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.conv = conv
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.attention = attention
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, trg):
        src, embed_src, src_mask = src
        trg, trg_mask = trg
        trg = self.layer_norm1(trg)
        k = self.conv.kernel_size
        trg = trg[:, k-1:] + self.conv(trg)
        trg = self.dropout1(trg)
        trg = trg.masked_fill(trg_mask.unsqueeze(-1)==0, 0)
        trg = self.layer_norm2(trg)
        trg = trg + self.attention(trg, src, embed_src, src_mask)
        trg = self.dropout2(trg)
        trg = trg.masked_fill(trg_mask.unsqueeze(-1)==0, 0)
        trg = self.layer_norm3(trg)
        trg = self.feed_forward(trg)
        trg = self.dropout3(trg)
        trg = trg.masked_fill(trg_mask.unsqueeze(-1)==0, 0)
        return trg