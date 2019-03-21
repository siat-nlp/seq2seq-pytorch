import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):

    def __init__(self, hidden_size, feed_forward_size, dropout=0.1):
        super().__init__()
        self.projection1 = nn.Linear(hidden_size, feed_forward_size)
        self.projection2 = nn.Linear(feed_forward_size, hidden_size)
        self.dropout = dropout

    def forward(self, x):
        output = self.projection2(F.relu(self.projection1(x)))
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output