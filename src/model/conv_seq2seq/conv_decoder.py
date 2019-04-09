import torch
from torch import nn
import torch.nn.functional as F
import math
from src.module.decoder import Decoder
from src.module.utils.constants import PAD_INDEX, SOS_INDEX
from src.module.utils.clone import clone

class ConvDecoder(Decoder):

    def __init__(self, embedding, positional_embedding, layer, num_layers, dropout, share_decoder_embedding=True):
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
        if share_decoder_embedding:
            self.generator.weight = embedding.weight

    def forward(self, src, trg):
        src, embed_src, src_mask = src
        return self.step(src, embed_src, src_mask, trg)

    def greedy_decode(self, src, max_len):
        src, embed_src, src_mask = src
        batch_size = src.size(0)
        trg = torch.zeros(batch_size, 1).fill_(SOS_INDEX).long().cuda()
        for i in range(max_len):
            logit = self.step(src, embed_src, src_mask, trg)[:, -1:]
            trg = torch.cat([trg, logit.argmax(dim=2, keepdim=False)], dim=1)
        trg = trg[:, 1:]
        return trg

    def beam_decode(self, src, max_len, beam_size):
        pass

    def step(self, src, embed_src, src_mask, trg):
        trg_embedding = self.embedding(trg) + self.positional_embedding(trg)
        trg_embedding = F.dropout(trg_embedding, p=self.dropout, training=self.training)
        trg_mask = (trg != PAD_INDEX).transpose(0, 1)
        trg_embedding = trg_embedding.transpose(0, 1)
        trg = trg_embedding
        trg = trg.masked_fill(trg_mask.unsqueeze(-1)==0, 0)
        trg = self.input_projection(trg)
        for layer in self.layers:
            trg = layer(src, embed_src, src_mask, trg, trg_embedding, trg_mask)
        trg = self.layer_norm(trg)
        trg = self.output_projection(trg)
        trg = trg.transpose(0, 1)
        logit = self.generator(trg)
        return logit

class ConvDecoderLayer(nn.Module):

    def __init__(self, hidden_size, embed_size, conv, attention, feed_forward, dropout):
        super(ConvDecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.conv = conv
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, embed_size)
        self.attention = attention
        self.output_projection = nn.Linear(embed_size, hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout = dropout

    def forward(self, src, embed_src, src_mask, trg, trg_embedding, trg_mask):
        trg = self.layer_norm1(trg)
        trg = trg + self.conv(trg)
        trg = F.dropout(trg, p=self.dropout, training=self.training)
        trg = trg.masked_fill(trg_mask.unsqueeze(-1)==0, 0)
        trg = self.layer_norm2(trg)
        query = (self.input_projection(trg) + trg_embedding).transpose(0, 1) * math.sqrt(0.5)
        context = self.output_projection(self.attention(query, src, embed_src, src_mask)).transpose(0, 1)
        trg = (trg + context) * math.sqrt(0.5)
        trg = F.dropout(trg, p=self.dropout, training=self.training)
        trg = trg.masked_fill(trg_mask.unsqueeze(-1)==0, 0)
        trg = self.layer_norm3(trg)
        trg = trg + self.feed_forward(trg)
        trg = F.dropout(trg, p=self.dropout, training=self.training)
        trg = trg.masked_fill(trg_mask.unsqueeze(-1)==0, 0)
        return trg