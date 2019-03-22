from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from src.module.encoder.encoder import Encoder
from src.module.utils.clone import clone
from src.module.utils.constants import PAD_INDEX
from src.module.layer.feed_forward import FeedForward
from src.module.layer.conv_glu import ConvGLU
from src.module.layer.conv_relu import ConvReLU

class ConvEncoder(Encoder):

    def __init__(self, embedding, positional_embedding, layer, num_layers, dropout):
        super(ConvEncoder, self).__init__()
        self.embedding = embedding
        self.positional_embedding = positional_embedding
        embed_size = embedding.embedding_dim
        hidden_size = layer.hidden_size
        self.input_projection = nn.Linear(embed_size, hidden_size)
        self.layers = clone(layer, num_layers)
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
        src = self.output_projection(src)
        src = src.transpose(0, 1)
        embed_src = (src + src_embedding) * math.sqrt(0.5)
        src = src.masked_fill(mask.unsqueeze(-1)==0, 0)
        embed_src = embed_src.masked_fill(mask.unsqueeze(-1)==0, 0)
        return src, embed_src, mask

class ConvEncoderLayer(nn.Module):

    def __init__(self, hidden_size, kernel_size, dropout, activatity='glu'):
        super(ConvEncoderLayer, self).__init__()
        self.layers = nn.ModuleList([
            nn.LayerNorm(hidden_size),
            ConvGLU(hidden_size, hidden_size, kernel_size) if activatity == 'glu'
            else ConvReLU(hidden_size, hidden_size, kernel_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            FeedForward(hidden_size, 2 * hidden_size),
            nn.Dropout(dropout)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x