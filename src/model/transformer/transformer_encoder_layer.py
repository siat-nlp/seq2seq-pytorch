from torch import nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size, attention, feed_forward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.attention = attention
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout = dropout

    def forward(self, src, src_mask):
        src = self.layer_norm1(src)
        src = src + self.attention(src, src, src, src_mask)
        src = F.dropout(src, p=self.dropout, training=self.training)
        src = self.layer_norm2(src)
        src = src + self.feed_forward(src)
        src = F.dropout(src, p=self.dropout, training=self.training)
        return src