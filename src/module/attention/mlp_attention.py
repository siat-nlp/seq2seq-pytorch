import torch
import torch.nn as nn
from torch.nn import init
from src.module.attention.attention import Attention

class MlpAttention(Attention):

    def __init__(self, query_size, key_size, out_size=1, dropout=0.1):
        super(MlpAttention, self).__init__(dropout)
        self.linear = nn.Linear(query_size + key_size, out_size, bias=False)
        self.v = nn.Parameter(torch.FloatTensor(out_size, 1))
        init.xavier_uniform_(self.v)

    def _score(self, query, key):
        time_step = key.size(1)
        batch_size, query_size = query.size()
        out_size = self.v.size(0)
        v = self.v.unsqueeze(0).expand(batch_size, out_size, 1)
        query = query.unsqueeze(1).expand(batch_size, time_step, query_size)    # (batch_size, time_step, query_size)
        score = torch.tanh(self.linear(torch.cat([query, key], dim=2))).matmul(v).unsqueeze(-1)
        return score