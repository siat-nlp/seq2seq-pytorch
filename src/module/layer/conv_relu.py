from torch import nn
import torch.nn.functional as F

class ConvReLU(nn.Module):

    def __init__(self, input_size, output_size, kernel_size):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        return F.relu(self.conv(x))