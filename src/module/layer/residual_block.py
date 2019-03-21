from torch import nn

class ResidualBlock(nn.Module):

    def __init__(self, layer):
        super(ResidualBlock, self).__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)

class ShiftResidualBlock(nn.Module):

    def __init__(self, layer, input_size, output_size):
        super(ShiftResidualBlock, self).__init__()
        self.layer = layer
        self.projection = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.projection(x) + self.layer(x)