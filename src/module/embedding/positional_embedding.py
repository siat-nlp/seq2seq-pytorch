import torch
from torch import nn
from torch.nn import init

class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super(LearnedPositionalEmbedding, self).__init__()
        self.embeddings = nn.Parameter(
            torch.FloatTensor(num_embeddings, embedding_dim)
        )
        init.normal_(self.embeddings, mean=0, std=0.1)
        init.constant_(self.embeddings[padding_idx], 0)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, x):
        mask = (x == self.padding_idx)
        batch_size, time_step = x.size()
        start = self.padding_idx + 1
        end = start + time_step
        x = torch.arange(start, end).unsqueeze(0).expand(batch_size, time_step).cuda()
        x = x.masked_fill(mask, self.padding_idx)
        x = x.view(-1)
        x = self.embeddings.index_select(dim=0, index=x)
        x = x.view(batch_size, time_step, self.embedding_dim)
        return x

class FixedPositionalEmbedding(nn.Module):

    def __init__(self):
        super(FixedPositionalEmbedding).__init__()

    def forward(self, *input):
        pass