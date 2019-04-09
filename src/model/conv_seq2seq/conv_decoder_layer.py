from torch import nn
import torch.nn.functional as F
import math

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