from torch import nn
import torch.nn.functional as F

class ConvGLU(nn.Module):
    """
    1-d convolutional layer using GLU activation.
    """

    def __init__(self, input_size, output_size, kernel_size, encode=True):
        super(ConvGLU, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=2 * output_size,
            kernel_size=kernel_size,
        )
        self.kernel_size = kernel_size
        self.left_padding = (kernel_size - 1) // 2 if encode else kernel_size - 1
        self.right_padding = kernel_size // 2 if encode else 0

    def forward(self, x):
        """
        :param x: FloatTensor (batch_size, time_step, input_size)
        :return: FloatTensor (batch_size, time_step, output_size)
        """
        x = x.transpose(1, 2)
        x = F.pad(x, [self.left_padding, self.right_padding, 0, 0, 0, 0])
        x = F.glu(self.conv(x), dim=1)
        x = x.transpose(1, 2)
        return x