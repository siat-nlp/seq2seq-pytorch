import torch.nn as nn
from src.module.encoder.encoder import Encoder

class TransformerEncoder(Encoder):

    def __init__(self):
        super(TransformerEncoder, self).__init__()

    def forward(self, src):
        pass

class EncoderLayer(nn.Module):

    def __init__(self):
        super(EncoderLayer, self).__init__()

    def forward(self, *input):
        pass