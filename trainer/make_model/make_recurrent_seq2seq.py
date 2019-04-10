from torch import nn
from src.model.recurrent_seq2seq.recurrent_seq2seq import RecurrentSeq2Seq
from src.model.recurrent_seq2seq.recurrent_encoder import RecurrentEncoder
from src.model.recurrent_seq2seq.recurrent_decoder import RecurrentDecoder
from data_process.utils import parse_path
import yaml

def make_recurrent_seq2seq(config):
    path = parse_path(config['data_process']['base_path'])
    data_log = yaml.load(open(path['log']['data_log']))
    share_src_trg_vocab = config['model']['share_src_trg_vocab']
    config = config['model'][config['model']['type']]
    if share_src_trg_vocab:
        src_embedding = nn.Embedding(
            num_embeddings=data_log['vocab_size'],
            embedding_dim=config['embed_size']
        )
        trg_embedding = src_embedding
    else:
        src_embedding = nn.Embedding(
            num_embeddings=data_log['src_vocab_size'],
            embedding_dim=config['embed_size']
        )
        trg_embedding = nn.Embedding(
            num_embeddings=data_log['trg_vocab_size'],
            embedding_dim=config['embed_size']
        )
    encoder = RecurrentEncoder(
        rnn_type=config['rnn_type'],
        embedding=src_embedding,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        bidirectional=config['bidirectional'],
        dropout=config['dropout']
    )
    decoder = RecurrentDecoder(
        rnn_type=config['rnn_type'],
        embedding=trg_embedding,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        share_decoder_embedding=config['share_decoder_embedding']
    )
    model = RecurrentSeq2Seq(encoder, decoder)
    return model