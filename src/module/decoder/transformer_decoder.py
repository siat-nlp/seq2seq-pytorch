import torch
from torch import nn
import torch.nn.functional as F
from src.module.decoder.decoder import Decoder
from src.module.utils.clone import clone
from src.module.utils.constants import PAD_INDEX, SOS_INDEX

class TransformerDecoder(Decoder):

    def __init__(self, embedding, positional_embedding, layer, num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = embedding
        self.positional_embedding = positional_embedding
        embed_size = embedding.embedding_dim
        vocab_size = embedding.num_embeddings
        hidden_size = layer.hidden_size
        # self.input_projection = nn.Linear(embed_size, hidden_size)
        self.layers = clone(layer, num_layers)
        # self.output_projection = nn.Linear(hidden_size, embed_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = dropout
        self.generator = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, trg):
        src, src_mask = src
        subsequent_mask = self.get_subsequent_mask(trg.size(1))
        trg_mask = (trg != PAD_INDEX)
        return self.step(src, src_mask, trg, trg_mask, subsequent_mask)

    def step(self, src, src_mask, trg, trg_mask, subsequent_mask):
        trg = self.embedding(trg) + self.positional_embedding(trg)
        trg = F.dropout(trg, p=self.dropout, training=self.training)
        for layer in self.layers:
            trg = layer(src, src_mask, trg, trg_mask, subsequent_mask)
        trg = self.layer_norm(trg)
        logit = self.generator(trg)
        return logit

    def greedy_decode(self, src, max_len):
        src, src_mask = src
        batch_size = src.size(0)
        subsequent_mask = self.get_subsequent_mask(max_len)
        trg_mask = torch.ones(batch_size, max_len).byte().cuda()
        trg = torch.zeros(batch_size, max_len)
        trg[:, 0] = SOS_INDEX
        for i in range(max_len):
            logit = self.step(src, src_mask, trg[:, i:i+1], trg_mask[:, i:i+1], subsequent_mask[i:i+1])
            logit = logit.argmax(dim=2, keepdim=False)
            if i < max_len - 1:
                trg[:, i + 1:i + 2] = logit

    def beam_decode(self, src, max_len, beam_size):
        pass

    def get_subsequent_mask(self, size):
        return torch.tril(torch.ones(size, size).byte().cuda())

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

    def forward(self, src, src_mask, trg, trg_mask, subsequent_mask):
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