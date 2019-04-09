import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from trainer.make_model.make_model import make_model
from trainer.make_data import make_train_data
from trainer.sentence_cross_entropy import SentenceCrossEntropy
from trainer.eval import eval

def train(config):
    model = make_model(config['model']).cuda()
    train_loader, val_loader = make_train_data(config)
    with open(config['data_process']['path']['processed']['trg_index2word'], 'rb') as handle:
        trg_index2word = pickle.load(handle)
    criterion = SentenceCrossEntropy()
    config = config['train']
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    for epoch in range(1, config['num_epoches'] + 1):
        sum_loss = 0
        sum_examples = 0
        s_loss = 0
        for i, data in enumerate(train_loader):
            src, trg = data
            src, trg = src.cuda(), trg.cuda()
            optimizer.zero_grad()
            logits = model(src, trg[:, 0:-1])
            loss = criterion(logits, trg[:, 1:])
            sum_loss += loss.item() * src.size(0)
            sum_examples += src.size(0)
            s_loss += loss.item()
            if i > 0 and i % 10 == 0:
                s_loss /= 10
                print('[epoch %2d] [step %4d] [loss %.4f]' % (epoch, i, s_loss))
                s_loss = 0
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            optimizer.step()
        avg_loss = sum_loss / sum_examples
        val_loss = eval(val_loader, model, config['max_len'], criterion, trg_index2word)
        print('[epoch %2d] [train loss %.4f] [val loss %.4f]' % (epoch, avg_loss, val_loss))