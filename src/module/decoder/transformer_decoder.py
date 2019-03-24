from torch import nn
from src.module.decoder.decoder import Decoder
from src.module.utils.clone import clone

class TransformerDecoder(Decoder):

    def __init__(self, embedding, positional_embedding, layer, num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = embedding
        self.positional_embedding = positional_embedding
        embed_size = embedding.embedding_dim
        hidden_size = layer.hidden_size
        self.input_projection = nn.Linear(embed_size, hidden_size)
        self.layers = clone(layer, num_layers)
        self.output_projection = nn.Linear(hidden_size, embed_size)
        self.dropout = dropout

    def forward(self, src):
        pass