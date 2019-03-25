from trainer.tensor2text import tensor2text

def eval(data_loader, model, max_len, index2word):
    texts = []
    for data in data_loader:
        src, trg = data
        src, trg = src.cuda(), trg.cuda()
        hyp = model.greedy_decode(src, max_len)
        texts.extend(tensor2text(hyp, index2word))
    for i in range(10):
        print(texts[i])