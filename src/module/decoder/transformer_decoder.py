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

    def forward(self, src, trg):
        pass

class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden_size, self_attention, src_attention, feed_forward, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attention = self_attention
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.src_attention = src_attention
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, trg):
        src, src_mask = src
        trg, trg_mask, subsequent_mask = trg
        trg = self.layer_norm1(trg)
        trg = trg + self.self_attention(trg, trg, trg, mask=trg_mask, subsequent_mask=subsequent_mask)
        trg = self.dropout1(trg)
        trg = self.layer_norm2(trg)
        trg = trg + self.src_attention(trg, src, src, mask=src_mask)
        trg = self.dropout2(trg)
        trg = self.layer_norm3(trg)
        trg = trg + self.feed_forward(trg)
        trg = self.dropout3(trg)
        return trg