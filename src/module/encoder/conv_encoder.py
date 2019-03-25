from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from src.module.encoder.encoder import Encoder
from src.module.utils.clone import clone
from src.module.utils.constants import PAD_INDEX

class ConvEncoder(Encoder):

    def __init__(self, embedding, positional_embedding, layer, num_layers, dropout):
        super(ConvEncoder, self).__init__()
        self.embedding = embedding
        self.positional_embedding = positional_embedding
        embed_size = embedding.embedding_dim
        hidden_size = layer.hidden_size
        self.input_projection = nn.Linear(embed_size, hidden_size)
        self.layers = clone(layer, num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, embed_size)
        self.dropout = dropout

    def forward(self, src):
        src_embedding = self.embedding(src) + self.positional_embedding(src)
        src_embedding = F.dropout(src_embedding, p=self.dropout, training=self.training)
        mask = (src != PAD_INDEX)   # ByteTensor (time_step, batch_size)
        src = src_embedding.transpose(0, 1) # (time_step, batch_size, embed_size)
        src = self.input_projection(src)
        for layer in self.layers:
            src = layer(src)
        src = self.layer_norm(src)
        src = self.output_projection(src)
        src = src.transpose(0, 1)
        embed_src = (src + src_embedding) * math.sqrt(0.5)
        src = src.masked_fill(mask.unsqueeze(-1)==0, 0)
        embed_src = embed_src.masked_fill(mask.unsqueeze(-1)==0, 0)
        return (src, embed_src, mask)

class ConvEncoderLayer(nn.Module):

    def __init__(self, hidden_size, conv, feed_forward, dropout):
        super(ConvEncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.conv = conv
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src = self.layer_norm1(src)
        src = self.conv(src)
        src = self.dropout1(src)
        src = self.layer_norm2(src)
        src = self.feed_forward(src)
        src = self.dropout2(src)
        return src