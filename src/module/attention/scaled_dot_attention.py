from src.module.attention.attention import Attention
import math

class ScaledDotAttention(Attention):

    def __init__(self, dropout=0.1):
        super(ScaledDotAttention, self).__init__(dropout)

    def _score(self, query, key):
        assert query.size(1) == key.size(2)
        return query.unsqueeze(1).matmul(key.transpose(1, 2)) / math.sqrt(query.size(1))