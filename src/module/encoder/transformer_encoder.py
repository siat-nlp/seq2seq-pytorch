from torch import nn
import torch.nn.functional as F
from src.module.encoder.encoder import Encoder
from src.module.utils.clone import clone

class TransformerEncoder(Encoder):

    def __init__(self, embedding, positional_embedding, layer, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
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


class EncoderLayer(nn.Module):

    def __init__(self):
        super(EncoderLayer, self).__init__()

    def forward(self, *input):
        pass