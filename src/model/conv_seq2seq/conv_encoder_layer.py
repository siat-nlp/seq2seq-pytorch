from torch import nn

class ConvEncoderLayer(nn.Module):

    def __init__(self, hidden_size, conv, feed_forward, dropout):
        super(ConvEncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.conv = conv
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = feed_forward
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src = self.layer_norm1(src)
        src = self.conv(src)
        src = self.dropout1(src)
        src = self.layer_norm2(src)
        src = self.feed_forward(src)
        src = self.dropout2(src)
        return src