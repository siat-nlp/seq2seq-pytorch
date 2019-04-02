from torch import nn
import torch.nn.functional as F

class FeedForward(nn.Module):

    def __init__(self, input_size, feed_forward_size, output_size):
        super(FeedForward, self).__init__()
        self.projection1 = nn.Linear(input_size, feed_forward_size)
        self.projection2 = nn.Linear(feed_forward_size, output_size)

    def forward(self, x):
        """
        x: FloatTensor (batch_size, time_step, hidden_size)
        """
        output = self.projection2(F.relu(self.projection1(x)))
        return output