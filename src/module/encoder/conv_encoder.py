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