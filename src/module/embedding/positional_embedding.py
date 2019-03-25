import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from src.module.utils.constants import PAD_INDEX

class PositionalEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, learnable=False):
        super(PositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.learnable = learnable
        if learnable:
            self.embeddings = nn.Parameter(
                torch.FloatTensor(num_embeddings, embedding_dim)
            )
            init.normal_(self.embeddings, mean=0, std=0.1)
        else:
            self.embeddings = torch.zeros(num_embeddings, embedding_dim)
            position = torch.arange(0, num_embeddings).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (- math.log(10000.0) / embedding_dim))
            self.embeddings[:, 0::2] = torch.sin(position * div_term)
            self.embeddings[:, 1::2] = torch.cos(position * div_term)
            self.embeddings = self.embeddings.cuda()

    def forward(self, x, left_padding=0):
        """
        x: LongTensor (batch_size, time_step)
        """
        x = x[:, left_padding:]
        x = self._get_embedding(x)
        x = F.pad(x, [0, 0, left_padding, 0, 0, 0])
        return x

    def _get_embedding(self, x):
        mask = (x == PAD_INDEX)
        batch_size, time_step = x.size()
        x = torch.arange(0, time_step).unsqueeze(0).expand(batch_size, time_step).cuda()
        x = x.view(-1)
        x = self.embeddings.index_select(dim=0, index=x)
        x = x.view(batch_size, time_step, self.embedding_dim)
        x = x.masked_fill(mask.unsqueeze(-1) == 0, 0)
        return x