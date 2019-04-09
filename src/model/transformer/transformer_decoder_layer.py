from torch import nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden_size, self_attention, src_attention, feed_forward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attention = self_attention
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.src_attention = src_attention
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout = dropout

    def forward(self, src, src_mask, trg, trg_mask):
        trg = self.layer_norm1(trg)
        trg = trg + self.self_attention(trg, trg, trg, mask=trg_mask)
        trg = F.dropout(trg, p=self.dropout, training=self.training)
        trg = self.layer_norm2(trg)
        trg = trg + self.src_attention(trg, src, src, mask=src_mask)
        trg = F.dropout(trg, p=self.dropout, training=self.training)
        trg = self.layer_norm3(trg)
        trg = trg + self.feed_forward(trg)
        trg = F.dropout(trg, p=self.dropout, training=self.training)
        return trg