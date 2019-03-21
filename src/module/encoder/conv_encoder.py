import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from src.module.encoder.encoder import Encoder
from src.module.utils.grad_multiply import GradMultiply

class ConvEncoder(Encoder):

    def __init__(self, embedding, hidden_size, kernel_size, num_layers, dropout):
        super(ConvEncoder, self).__init__()
        assert kernel_size % 2 == 1
        self.embedding = embedding
        self.dropout = dropout
        embed_size = embedding.embedding_dim
        self.input_projection = nn.Linear(embed_size, hidden_size)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=2 * hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * hidden_size))
            nn.init.normal_(conv.weight, mean=0, std=std)
            init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.convs.append(conv)
        self.output_projection = nn.Linear(hidden_size, embed_size)

    def forward(self, src):
        src_embedding = self.embedding(src)
        src_embedding = F.dropout(src_embedding, p=self.dropout, training=self.training)
        mask = (src != 0).transpose(0, 1)   # ByteTensor (time_step, batch_size)
        src = src_embedding.transpose(0, 1) # (time_step, batch_size, embed_size)
        src = self.input_projection(src)
        for conv in self.convs:
            src = src.masked_fill(mask.unsqueeze(-1), 0)
            residual = src
            src = F.dropout(src, p=self.dropout, training=self.training)
            src = conv(src)
            src = F.glu(src, dim=2)
            src = (src + residual) * math.sqrt(0.5)
        src = self.output_projection(src)
        src = src.masked_fill(mask.unsqueeze(-1), 0)
        src = src.transpose(0, 1)
        # src = GradMultiply(src, 1.0 / (2.0 * self.num_attention_layers))
        embed_src = (src + src_embedding) * math.sqrt(0.5)
        return src, embed_src, mask