from torch import nn

class SublayerConnection(nn.Module):

    def __init__(self, layer, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(layer.hidden_size)
        self.layer = layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.layer(self.layer_norm(x)))