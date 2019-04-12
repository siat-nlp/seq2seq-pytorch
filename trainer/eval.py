import torch
from trainer.tensor2text import tensor2text
import random

def eval(data_loader, model, max_len, criterion, index2word):
    total_loss = 0
    total_samples = 0
    texts = []
    for data in data_loader:
        src, trg = data
        src, trg = src.cuda(), trg.cuda()
        with torch.no_grad():
            logit = model.decode(src, max_len)
        time_step = trg.size(1)
        logit = logit[:, 0:time_step]
        batch_size = src.size(0)
        total_loss += batch_size * criterion(logit, trg)
        total_samples += batch_size
        hyp = logit.argmax(dim=-1)
        texts.extend(tensor2text(hyp, index2word))
    loss = total_loss / total_samples
    random.shuffle(texts)
    for i in range(10):
        print(texts[i])
    return loss