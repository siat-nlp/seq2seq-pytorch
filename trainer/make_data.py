from torch.utils.data import DataLoader
from data_process.dataset import Seq2SeqDataset

def make_train_data(config):
    train_dataset = Seq2SeqDataset(config['data_process']['path']['processed']['train'])
    val_dataset = Seq2SeqDataset(config['data_process']['path']['processed']['val'])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    return train_loader, val_loader

def make_test_data(config):
    test_dataset = Seq2SeqDataset(config['data_process']['path']['processed']['test'])
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    return test_loader