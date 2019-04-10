import numpy as np
from src.module.utils.constants import UNK, SOS, EOS

def text_file2word_lists(text_file, tokenizer):
    word_lists = []
    for text in text_file.readlines():
        word_lists.append([SOS] + tokenizer(text.strip()) + [EOS])
    return word_lists

def word_lists2numpy(word_lists, word2index):
    num = len(word_lists)
    max_len = 0
    for word_list in word_lists:
        max_len = max(max_len, len(word_list))
    max_index = 0
    min_index = 30000
    for i in range(num):
        word_lists[i] = list(map(lambda x: word2index[x] if x in word2index else word2index[UNK], word_lists[i]))
        word_lists[i].extend([0] * (max_len - len(word_lists[i])))
        max_index = max(max_index, max(word_lists[i]))
        min_index = min(min_index, min(word_lists[i]))
    print(max_index, min_index)
    return np.array(word_lists)