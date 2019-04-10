import os
import yaml
import numpy as np
import pickle
from data_process.vocab import Vocab
from data_process.utils import get_word_lists, word_lists2numpy, parse_path

def data_process(config):
    path = parse_path(config['data_process']['base_path'])
    src_train_word_lists = get_word_lists(path['raw']['src_train'])
    trg_train_word_lists = get_word_lists(path['raw']['trg_train'])
    src_val_word_lists = get_word_lists(path['raw']['src_val'])
    trg_val_word_lists = get_word_lists(path['raw']['trg_val'])
    src_test_word_lists = get_word_lists(path['raw']['src_test'])
    trg_test_word_lists = get_word_lists(path['raw']['trg_test'])

    if config['share_src_trg_vocab']:
        vocab = Vocab()

        for word_list in src_train_word_lists:
            vocab.add_list(word_list)
        for word_list in trg_train_word_lists:
            vocab.add_list(word_list)

        word2index, index2word = vocab.get_vocab(max_size=config['data_process']['vocab']['max_size'],
                                                 min_freq=config['data_process']['vocab']['min_freq'])
        src_train = word_lists2numpy(src_train_word_lists, word2index)
        trg_train = word_lists2numpy(trg_train_word_lists, word2index)
        src_val = word_lists2numpy(src_val_word_lists, word2index)
        trg_val = word_lists2numpy(trg_val_word_lists, word2index)
        src_test = word_lists2numpy(src_test_word_lists, word2index)
        trg_test = word_lists2numpy(trg_test_word_lists, word2index)

        if not os.path.exists(os.path.dirname(path['processed']['src_train'])):
            os.makedirs(os.path.dirname(path['processed']['src_train']))

        np.savez(path['processed']['train'], src=src_train, trg=trg_train)
        np.savez(path['processed']['val'], src=src_val, trg=trg_val)
        np.savez(path['processed']['test'], src=src_test, trg=trg_test)

        with open(path['processed']['word2index'], 'wb') as handle:
            pickle.dump(word2index, handle)
        with open(path['processed']['index2word'], 'wb') as handle:
            pickle.dump(index2word, handle)
        data_log = {
            'vocab_size': len(index2word),
            'oov_size': len(word2index) - len(index2word),
            'train_src_max_len': src_train.shape[1],
            'train_trg_max_len': trg_train.shape[1],
            'val_src_max_len': src_val.shape[1],
            'val_trg_max_len': trg_val.shape[1],
            'test_src_max_len': src_test.shape[1],
            'test_trg_max_len': trg_test.shape[1]
        }
        if not os.path.exists(os.path.dirname(path['log']['data_log'])):
            os.makedirs(os.path.dirname(path['log']['data_log']))
        with open(path['log']['data_log'], 'w') as handle:
            yaml.safe_dump(data_log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)
    else:
        src_vocab = Vocab()
        trg_vocab = Vocab()

        for word_list in src_train_word_lists:
            src_vocab.add_list(word_list)
        for word_list in trg_train_word_lists:
            trg_vocab.add_list(word_list)

        src_word2index, src_index2word = src_vocab.get_vocab(
            max_size=config['data_process']['vocab']['max_size'],
            min_freq=config['data_process']['vocab']['min_freq']
        )
        trg_word2index, trg_index2word = trg_vocab.get_vocab(
            max_size=config['data_process']['vocab']['max_size'],
            min_freq=config['data_process']['vocab']['min_freq']
        )

        src_train = word_lists2numpy(src_train_word_lists, src_word2index)
        trg_train = word_lists2numpy(trg_train_word_lists, trg_word2index)
        src_val = word_lists2numpy(src_val_word_lists, src_word2index)
        trg_val = word_lists2numpy(trg_val_word_lists, trg_word2index)
        src_test = word_lists2numpy(src_test_word_lists, src_word2index)
        trg_test = word_lists2numpy(trg_test_word_lists, trg_word2index)

        if not os.path.exists(os.path.dirname(path['processed']['src_train'])):
            os.makedirs(os.path.dirname(path['processed']['src_train']))

        np.savez(path['processed']['train'], src=src_train, trg=trg_train)
        np.savez(path['processed']['val'], src=src_val, trg=trg_val)
        np.savez(path['processed']['test'], src=src_test, trg=trg_test)

        with open(path['processed']['src_word2index'], 'wb') as handle:
            pickle.dump(src_word2index, handle)
        with open(path['processed']['src_index2word'], 'wb') as handle:
            pickle.dump(src_index2word, handle)
        with open(path['processed']['trg_word2index'], 'wb') as handle:
            pickle.dump(trg_word2index, handle)
        with open(path['processed']['trg_index2word'], 'wb') as handle:
            pickle.dump(trg_index2word, handle)
        data_log = {
            'src_vocab_size': len(src_index2word),
            'src_oov_size': len(src_word2index) - len(src_index2word),
            'trg_vocab_size': len(trg_index2word),
            'trg_oov_size': len(trg_word2index) - len(trg_index2word),
            'train_src_max_len': src_train.shape[1],
            'train_trg_max_len': trg_train.shape[1],
            'val_src_max_len': src_val.shape[1],
            'val_trg_max_len': trg_val.shape[1],
            'test_src_max_len': src_test.shape[1],
            'test_trg_max_len': trg_test.shape[1]
        }
        if not os.path.exists(os.path.dirname(path['log']['data_log'])):
            os.makedirs(os.path.dirname(path['log']['data_log']))
        with open(path['log']['data_log'], 'w') as handle:
            yaml.safe_dump(data_log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)