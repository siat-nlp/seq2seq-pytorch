import torch
from torch.utils.data import Dataset
import numpy as np

class Seq2SeqDataset(Dataset):

    def __init__(self, path):
        super(Seq2SeqDataset, self).__init__()
        data = np.load(path)
        self.src = torch.tensor(data['src']).long()
        self.trg = torch.tensor(data['trg']).long()
        self.len = self.src.size(0)

    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return self.len