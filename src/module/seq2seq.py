import torch.nn as nn

class Seq2Seq(nn.Module):
    """
    The base class of all seq2seq models.
    """

    def __init__(self):
        super(Seq2Seq, self).__init__()

    def forward(self, src, trg):
        raise NotImplementedError('Seq2Seq forward method is not implemented.')