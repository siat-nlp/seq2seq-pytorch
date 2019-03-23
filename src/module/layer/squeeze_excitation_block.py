from torch import nn
import torch.nn.functional as F

class SqueezeExcitationBlock(nn.Module):

    def __init__(self, hidden_size, squeeze_size, batch_first=True):
        super(SqueezeExcitationBlock, self).__init__()
        self.projection1 = nn.Linear(hidden_size, squeeze_size)
        self.projection2 = nn.Linear(squeeze_size, hidden_size)
        self.batch_first = batch_first

    def forward(self, x):
        """
        x: FloatTensor (batch_size, time_step, hidden_size)
        """
        y = x.mean(dim=1 if self.batch_first else 0, keepdim=True)
        y = F.sigmoid(self.projection2(F.relu(self.projection1(y))))
        x = x * y
        return x