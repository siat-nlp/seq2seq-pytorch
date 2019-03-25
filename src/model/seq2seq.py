import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self):
        super(Seq2Seq, self).__init__()

    def forward(self, src, trg):
        raise NotImplementedError('Seq2Seq forward method is not implemented.')

    def greedy_decode(self, src):
        raise NotImplementedError('Seq2Seq forward method is not implemented.')

    def beam_decode(self, src):
        raise NotImplementedError('Seq2Seq forward method is not implemented.')