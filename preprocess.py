import os
import yaml
import numpy as np
import pickle
from data_process.vocab import Vocab
from data_process.tokenizer import fair_tokenizer, nltk_tokenizer, spacy_en_tokenizer, spacy_de_tokenizer
from data_process.utils import text_file2word_lists, word_lists2numpy

config = yaml.load(open('configs/config.yml'))['data_process']

if config['tokenizer'] == 'fair':
    src_tokenizer = fair_tokenizer
    trg_tokenizer = fair_tokenizer
elif config['tokenizer'] == 'nltk':
    src_tokenizer = nltk_tokenizer
    trg_tokenizer = nltk_tokenizer
elif config['tokenizer'] == 'spacy':
    src_tokenizer = spacy_de_tokenizer
    trg_tokenizer = spacy_en_tokenizer
else:
    raise ValueError('No supporting.')

src_train_text = open(config['path']['raw']['src_train'], 'r', encoding='utf-8')
trg_train_text = open(config['path']['raw']['trg_train'], 'r', encoding='utf-8')
src_val_text = open(config['path']['raw']['src_val'], 'r', encoding='utf-8')
trg_val_text = open(config['path']['raw']['trg_val'], 'r', encoding='utf-8')
src_test_text = open(config['path']['raw']['src_test'], 'r', encoding='utf-8')
trg_test_text = open(config['path']['raw']['trg_test'], 'r', encoding='utf-8')

src_train_word_lists = text_file2word_lists(src_train_text, src_tokenizer)
trg_train_word_lists = text_file2word_lists(trg_train_text, trg_tokenizer)
src_val_word_lists = text_file2word_lists(src_val_text, src_tokenizer)
trg_val_word_lists = text_file2word_lists(trg_val_text, trg_tokenizer)
src_test_word_lists = text_file2word_lists(src_test_text, src_tokenizer)
trg_test_word_lists = text_file2word_lists(trg_test_text, trg_tokenizer)

src_vocab = Vocab()
trg_vocab = Vocab()

for word_list in src_train_word_lists:
    src_vocab.add_list(word_list)
for word_list in trg_train_word_lists:
    trg_vocab.add_list(word_list)

src_word2index, src_index2word = src_vocab.get_vocab(
    max_size=config['vocab']['src']['max_size'],
    min_freq=config['vocab']['src']['min_freq']
)
trg_word2index, trg_index2word = trg_vocab.get_vocab(
    max_size=config['vocab']['trg']['max_size'],
    min_freq=config['vocab']['trg']['min_freq']
)

src_train = word_lists2numpy(src_train_word_lists, src_word2index)
trg_train = word_lists2numpy(trg_train_word_lists, trg_word2index)
src_val = word_lists2numpy(src_val_word_lists, src_word2index)
trg_val = word_lists2numpy(trg_val_word_lists, trg_word2index)
src_test = word_lists2numpy(src_test_word_lists, src_word2index)
trg_test = word_lists2numpy(trg_test_word_lists, trg_word2index)

if not os.path.exists(os.path.dirname(config['path']['processed']['train'])):
    os.makedirs(os.path.dirname(config['path']['processed']['train']))

np.savez(config['path']['processed']['train'], src=src_train, trg=trg_train)
np.savez(config['path']['processed']['val'], src=src_val, trg=trg_val)
np.savez(config['path']['processed']['test'], src=src_test, trg=trg_test)

with open(config['path']['processed']['src_word2index'], 'wb') as handle:
    pickle.dump(src_word2index, handle)
with open(config['path']['processed']['src_index2word'], 'wb') as handle:
    pickle.dump(src_index2word, handle)
with open(config['path']['processed']['trg_word2index'], 'wb') as handle:
    pickle.dump(trg_word2index, handle)
with open(config['path']['processed']['trg_index2word'], 'wb') as handle:
    pickle.dump(trg_index2word, handle)