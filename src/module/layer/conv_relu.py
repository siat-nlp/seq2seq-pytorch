from torch import nn
import torch.nn.functional as F

class ConvReLU(nn.Module):

    def __init__(self, input_size, output_size, kernel_size, encode=True):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=kernel_size,
        )
        self.kernel_size = kernel_size
        self.left_padding = (kernel_size - 1) // 2 if encode else 0
        self.right_padding = kernel_size // 2 if encode else 0

    def forward(self, x):
        """
        x: FloatTensor (batch_size, time_step, hidden_size)
        """
        x = x.transpose(1, 2)
        x = F.pad(x, [self.left_padding, self.right_padding, 0, 0, 0, 0])
        x = F.relu(self.conv(x))
        x = x.transpose(1, 2)
        return x