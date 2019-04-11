from torch import nn
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
        :param query: FloatTensor (batch_size, query_size) or FloatTensor (batch_size, num_queries, query_size)
        :param key: FloatTensor (batch_size, time_step, key_size)
        :param value: FloatTensor (batch_size, time_step, hidden_size)
        :param mask: ByteTensor (batch_size, time_step) or ByteTensor (batch_size, num_queries, time_step) or None
        :return output: FloatTensor (batch_size, query_size) or FloatTensor (batch_size, num_queries, query_size)
        """
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1)
            else:
                assert mask.size(1) == query.size(1)
        score = self._score(query, key) # FloatTensor (batch_size, num_queries, time_step)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = weights.matmul(value)
        if single_query:
            output = output.squeeze(1)
        return output

    def _score(self, query, key):
        """
        :param query: FloatTensor (batch_size, num_queries, query_size)
        :param key: FloatTensor (batch_size, time_step, key_size)
        :return score: FloatTensor (batch_size, num_queries, time_step)
        """
        raise NotImplementedError('Attention score method is not implemented.')

    def _weights_normalize(self, score, mask):
        """
        :param score: FloatTensor (batch_size, num_queries, time_step)
        :param mask: ByteTensor (batch_size, num_queries, time_step) or (batch_size, 1, time_step) or None
        :return weights: FloatTensor (batch_size, num_queries, time_step)
        """
        if not mask is None:
            score = score.masked_fill(mask == 0, -constants.INF)
        weights = F.softmax(score, dim=-1)
        return weights

    def get_attention_weights(self, query, key, mask=None):
        """
        :param query: FloatTensor (batch_size, query_size) or FloatTensor (batch_size, num_queries, query_size)
        :param key: FloatTensor (batch_size, time_step, key_size)
        :param mask: ByteTensor (batch_size, time_step) or ByteTensor (batch_size, num_queries, time_step) or None
        :return weights: FloatTensor (batch_size, time_step) or FloatTensor (batch_size, num_queries, time_step)
        """
        single_query = False
        if len(query.size()) == 2:
            query = query.unsqueeze(1)
            single_query = True
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1)
            else:
                assert mask.size(1) == query.size(1)
        score = self._score(query, key)  # FloatTensor (batch_size, num_queries, time_step)
        weights = self._weights_normalize(score, mask)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if single_query:
            weights = weights.squeeze(1)
        return weights