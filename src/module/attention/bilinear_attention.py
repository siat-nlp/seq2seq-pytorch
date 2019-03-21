import torch
import torch.nn as nn
from torch.nn import init
from src.module.attention.attention import Attention

class BilinearAttention(Attention):

    def __init__(self, query_size, key_size, dropout=0.1):
        super(BilinearAttention, self).__init__(dropout)
        self.weights = nn.Parameter(torch.FloatTensor(key_size, query_size))
        init.xavier_uniform_(self.weights)

    def _score(self, query, key):
        query_size = query.size(1)
        batch_size, time_step, key_size = key.size()
        weights = self.weights.expand(batch_size, key_size, query_size)  # (batch_size, key_size, query_size)
        query = query.unsqueeze(-1)  # (batch_size, query_size, 1)
        mids = weights.matmul(query)  # (batch_size, key_size, 1)
        mids = mids.unsqueeze(1).expand(batch_size, time_step, key_size, 1)  # (batch_size, time_step, key_size, 1)
        key = key.unsqueeze(-2)  # (batch_size, time_step, 1, key_size)
        scores = key.matmul(mids).squeeze(-1).transpose(1, 2)  # (batch_size, time_step)
        return scores