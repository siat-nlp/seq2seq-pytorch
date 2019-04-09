from trainer.tensor2text import tensor2text

def eval(data_loader, model, max_len, criterion, index2word):
    total_loss = 0
    total_samples = 0
    for data in data_loader:
        src, trg = data
        src, trg = src.cuda(), trg.cuda()
        logit = model.greedy_decode(src, max_len)
        batch_size = src.size(0)
        time_step = trg.size(1)
        total_loss += batch_size * criterion(logit[:, 0:time_step], trg)
        total_samples += batch_size
    loss = total_loss / total_samples
    return loss