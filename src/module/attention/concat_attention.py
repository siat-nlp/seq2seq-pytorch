import torch
import torch.nn as nn
from src.module.attention.attention import Attention

class ConcatAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0.1):
        super(ConcatAttention, self).__init__(dropout)
        self.projection = nn.Linear(query_size + key_size, 1)

    def _score(self, query, key):
        time_step = key.size(1)
        batch_size, query_size = query.size()
        query = query.unsqueeze(1).expand(batch_size, time_step, query_size)  # (batch_size, time_step, query_size)
        scores = self.projection(torch.cat([query, key], dim=2)).transpose(1, 2)
        return scores