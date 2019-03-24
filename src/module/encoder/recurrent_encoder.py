import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.encoder.encoder import Encoder

class RecurrentEncoder(Encoder):

    def __init__(self, embedding, layer, num_layers, dropout):
        super(RecurrentEncoder, self).__init__()
        self.embedding = embedding
        self.dropout = dropout

    def forward(self, src):
        src_embedding = self.embedding(src)
        src_embedding = F.dropout(src_embedding, p=self.dropout, training=self.training)
        

class RecurrentEncoderLayer(nn.Module):

    def __init__(self):
        super(RecurrentEncoderLayer, self).__init__()

    def forward(self, *input):
        pass