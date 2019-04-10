from torch.utils.data import DataLoader
from data_process.dataset import Seq2SeqDataset
from data_process.utils import parse_path

def make_train_data(config):
    path = parse_path(config['data_process']['base_path'])
    train_dataset = Seq2SeqDataset(path['processed']['train'])
    val_dataset = Seq2SeqDataset(path['processed']['val'])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['model'][config['model']['type']]['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    return train_loader, val_loader

def make_test_data(config):
    path = parse_path(config['data_process']['base_path'])
    test_dataset = Seq2SeqDataset(path['processed']['test'])
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    return test_loader