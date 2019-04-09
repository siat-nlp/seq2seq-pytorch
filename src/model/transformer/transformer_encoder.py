from torch import nn
import torch.nn.functional as F
from src.module.encoder import Encoder
from src.module.utils.clone import clone
from src.module.utils.constants import PAD_INDEX

class TransformerEncoder(Encoder):

    def __init__(self, embedding, positional_embedding, layer, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = embedding
        self.positional_embedding = positional_embedding
        # embed_size = embedding.embedding_dim
        hidden_size = layer.hidden_size
        # self.input_projection = nn.Linear(embed_size, hidden_size)
        self.layers = clone(layer, num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # self.output_projection = nn.Linear(hidden_size, embed_size)
        self.dropout = dropout

    def forward(self, src):
        src_embedding = self.embedding(src) + self.positional_embedding(src)
        src_embedding = F.dropout(src_embedding, p=self.dropout, training=self.training)
        src_mask = (src != PAD_INDEX)
        src = src_embedding
        for layer in self.layers:
            src = layer(src, src_mask)
        src = self.layer_norm(src)
        return (src, src_mask)