import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from src.module.utils import constants

class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, key_size, value_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.query_projection = nn.Linear(hidden_size, num_heads * key_size)
        self.key_projection = nn.Linear(hidden_size, num_heads * key_size)
        self.value_projection = nn.Linear(hidden_size, num_heads * value_size)
        init.normal_(self.query_projection.weight, mean=0, std=math.sqrt(2.0 / (hidden_size + key_size)))
        init.normal_(self.key_projection.weight, mean=0, std=math.sqrt(2.0 / (hidden_size + key_size)))
        init.normal_(self.value_projection.weight, mean=0, std=math.sqrt(2.0 / (hidden_size + value_size)))
        self.output_projection = nn.Linear(num_heads * value_size, hidden_size)
        init.xavier_normal_(self.output_projection.weight)
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        """
        query: FloatTensor (batch_size, time_step, hidden_size)
        key: FloatTensor (batch_size, time_step, hidden_size)
        value: FloatTensor (batch_size, time_step, hidden_size)
        mask: ByteTensor (batch_size, time_step)
        """
        num_heads, key_size, value_size = self.num_heads, self.key_size, self.value_size
        batch_size, time_step, _ = query.size()
        query = self.query_projection(query).view(batch_size, time_step, num_heads, key_size)
        key = self.key_projection(key).view(batch_size, time_step, num_heads, key_size)
        value = self.value_projection(value).view(batch_size, time_step, num_heads, value_size)
        mask = mask.unsqueeze(0).expand(num_heads, batch_size, time_step).view(-1, time_step)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, time_step, key_size)
        key = key.permute(2, 0, 3, 1).contiguous().view(-1, key_size, time_step)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, time_step, value_size)
        score = query.matmul(key)
        score = score.masked_fill(mask.unsqueeze(1)==0, -constants.INF)
        probability = F.softmax(score, dim=-1)
        probability = F.dropout(probability, p=self.dropout, training=self.training)
        output = probability.matmul(value)
        output = output.masked_fill(mask.unsqueeze(-1)==0, 0)
        output = output.view(num_heads, batch_size, time_step, value_size)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, time_step, -1)
        output = self.output_projection(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output