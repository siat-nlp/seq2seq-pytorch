import os
import numpy as np
from src.module.utils.constants import UNK, SOS, EOS
from data_process.tokenizer import fair_tokenizer

def get_word_lists(path):
    text_file = open(path, 'r', encoding='utf-8')
    word_lists = []
    for text in text_file.readlines():
        word_lists.append([SOS] + fair_tokenizer(text.strip()) + [EOS])
    return word_lists

def analyze(word_lists):
    pass

def word_lists2numpy(word_lists, word2index):
    num = len(word_lists)
    max_len = 0
    for word_list in word_lists:
        max_len = max(max_len, len(word_list))
    for i in range(num):
        word_lists[i] = list(map(lambda x: word2index[x] if x in word2index else word2index[UNK], word_lists[i]))
        word_lists[i].extend([0] * (max_len - len(word_lists[i])))
    return np.array(word_lists)

def parse_path(base_path):
    return {
        'raw': {
            'src_train': os.path.join(base_path, 'raw/src_train.txt'),
            'trg_train': os.path.join(base_path, 'raw/trg_train.txt'),
            'src_val': os.path.join(base_path, 'raw/src_val.txt'),
            'trg_val': os.path.join(base_path, 'raw/trg_val.txt'),
            'src_test': os.path.join(base_path, 'raw/src_test.txt'),
            'trg_test': os.path.join(base_path, 'raw/trg_test.txt')
        },
        'processed': {
            'train': os.path.join(base_path, 'processed/train.npz'),
            'val': os.path.join(base_path, 'processed/val.npz'),
            'test': os.path.join(base_path, 'processed/test.npz'),
            'src_word2index': os.path.join(base_path, 'processed/src_word2index.pickle'),
            'src_index2word': os.path.join(base_path, 'processed/src_index2word.pickle'),
            'trg_word2index': os.path.join(base_path, 'processed/trg_word2index.pickle'),
            'trg_index2word': os.path.join(base_path, 'processed/trg_index2word.pickle'),
            'word2index': os.path.join(base_path, 'processed/word2index.pickle'),
            'index2word': os.path.join(base_path, 'processed/index2word.pickle')
        },
        'log': {
            'data_log': os.path.join(base_path, 'log/data_log.yml')
        }
    }