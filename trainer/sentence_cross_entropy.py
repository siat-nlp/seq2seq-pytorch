import torch
from torch import nn
import torch.nn.functional as F
from src.module.utils.constants import PAD_INDEX

class SentenceCrossEntropy(nn.Module):

    def __init__(self, label_smoothing=0.1):
        super(SentenceCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, hyp, ref):
        """
        :param hyp: FloatTensor (batch_size, time_step, vocab_size)
        :param ref: LongTensor (batch_size, time_step)
        :return: loss function value
        """
        vocab_size = hyp.size(2)
        hyp = hyp.view(-1, vocab_size)
        ref = ref.view(-1)
        mask = (ref != PAD_INDEX)
        one_hot = torch.zeros_like(hyp).scatter(1, ref, 1).float().cuda()
        ref = one_hot * (1.0 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (vocab_size - 1)
        log_prob = F.log_softmax(hyp, dim=1)
        loss = - (log_prob * ref).sum(dim=1)
        loss = loss.masked_select(mask).mean()
        return loss