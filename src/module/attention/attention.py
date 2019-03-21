import torch.nn as nn
import torch.nn.functional as F
from src.module.utils import constants

class Attention(nn.Module):
    """
    The base class of attention.
    """

    def __init__(self, dropout):
        super(Attention, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        """
        query: FloatTensor (batch_size, query_size)
        key: FloatTensor (batch_size, time_step, key_size)
        value: FloatTensor (batch_size, time_step, hidden_size)
        mask: ByteTensor (batch_size, time_step)
        """
        score = self._score(query, key)
        probability = self._probability_normalize(score, mask)
        probability = F.dropout(probability, p=self.dropout, training=self.training)
        output = self._attention_aggregate(probability, value)
        return output

    def _score(self, query, key):
        raise NotImplementedError('Attention score method is not implemented.')

    def _probability_normalize(self, score, mask):
        if not mask is None:
            score = score.masked_fill(mask.unsqueeze(1) == 0, -constants.INF)
        probability = F.softmax(score, dim=-1)
        return probability

    def _attention_aggregate(self, probability, value):
        return probability.matmul(value).squeeze(1)