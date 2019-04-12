import torch
from torch import nn
import torch.nn.functional as F
from src.module.decoder import Decoder
from src.module.utils.clone import clone
from src.module.utils.constants import PAD_INDEX, SOS_INDEX

class TransformerDecoder(Decoder):

    def __init__(self, embedding, positional_embedding, layer, num_layers, dropout, share_decoder_embedding=True):
        super(TransformerDecoder, self).__init__()
        self.embedding = embedding
        self.positional_embedding = positional_embedding
        embed_size = embedding.embedding_dim
        vocab_size = embedding.num_embeddings
        hidden_size = layer.hidden_size
        assert embed_size == hidden_size
        # self.input_projection = nn.Linear(embed_size, hidden_size)
        self.layers = clone(layer, num_layers)
        # self.output_projection = nn.Linear(hidden_size, embed_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = dropout
        self.generator = nn.Linear(hidden_size, vocab_size)
        if share_decoder_embedding:
            self.generator.weight = embedding.weight

    def forward(self, src, trg):
        src, src_mask = src
        trg_mask = self.get_mask(trg)
        return self.step(src, src_mask, trg, trg_mask)

    def step(self, src, src_mask, trg, trg_mask):
        trg = self.embedding(trg) + self.positional_embedding(trg)
        trg = F.dropout(trg, p=self.dropout, training=self.training)
        for layer in self.layers:
            trg = layer(src, src_mask, trg, trg_mask)
        trg = self.layer_norm(trg)
        logit = self.generator(trg)
        return logit

    def decode(self, src, max_len):
        src, src_mask = src
        batch_size = src.size(0)
        trg = torch.zeros(batch_size, 1).fill_(SOS_INDEX).long().cuda()
        logit = []
        for i in range(max_len):
            trg_mask = self.get_mask(trg)
            step_logit = self.step(src, src_mask, trg, trg_mask)[:, -1:]
            trg = torch.cat([trg, step_logit.argmax(dim=2, keepdim=False)], dim=1)
            logit.append(step_logit.squeeze(1))
        logit = torch.stack(logit, dim=1)
        return logit

    def get_mask(self, trg):
        """
        trg: LongTensor (batch_size, time_step)
        """
        batch_size, time_step = trg.size()
        trg_mask = (trg != PAD_INDEX).unsqueeze(1).expand(batch_size, time_step, time_step)
        subsequent_mask = torch.tril(torch.ones(time_step, time_step).byte().cuda())
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, time_step, time_step)
        trg_mask = trg_mask & subsequent_mask
        return trg_mask