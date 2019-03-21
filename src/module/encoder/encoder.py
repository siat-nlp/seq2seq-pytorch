import torch.nn as nn

class Encoder(nn.Module):
    """
    Base class for all encoders.
    """

    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, src):
        """
        src: LongTensor (batch_size, time_step)
        """
        raise NotImplementedError('Encoder forward method is not implemented.')