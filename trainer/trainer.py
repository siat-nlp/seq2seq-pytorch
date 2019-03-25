import torch.nn as nn
import torch.optim as optim
from trainer.make_model.make_model import make_model
from trainer.make_data import make_train_data
from trainer.masked_cross_entropy import masked_cross_entropy

def train(config):
    model = make_model(config['model']).cuda()
    train_loader, val_loader = make_train_data(config)
    criterion = nn.CrossEntropyLoss(reduction='none')
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
            loss = masked_cross_entropy(logits, trg[:, 1:], criterion)
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
        print('[epoch %2d] [loss %.4f]' % (epoch, avg_loss))