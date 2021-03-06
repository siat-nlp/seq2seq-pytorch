from torch import nn

class Decoder(nn.Module):
    """
    The base class for all decoders.
    """

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, src, trg):
        raise NotImplementedError('Decoder forward method is not implemented.')

    def decode(self, src, max_len):
        raise NotImplementedError('Decoder greedy_decode method is not implemented.')