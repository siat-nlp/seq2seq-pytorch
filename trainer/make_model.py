import torch.nn as nn
from src.module.encoder.recurrent_encoder import RecurrentEncoder
from src.module.utils import Bridge
from src.module.decoder.recurrent_decoder import RecurrentDecoder
from src.model.recurrent_seq2seq import RecurrentSeq2Seq
from src.module.utils import MultiLayerLSTMCells
from src.module.utils.rnn_cell.multi_layer_gru_cells import MultiLayerGRUCells
from src.module.attention.bilinear_attention import BilinearAttention

def make_model(config):
    if config['type'] == 'rnn':
        return make_recurrent_seq2seq(config['rnn'])
    elif config['type'] == 'cnn':
        return make_conv_seq2seq(config['cnn'])
    elif config['type'] == 'transformer':
        return make_transformer(config['transformer'])
    else:
        raise ValueError('No supporting.')

def make_recurrent_seq2seq(config):
    src_embedding = make_embedding(config['src_embedding'])
    trg_embedding = make_embedding(config['trg_embedding'])
    encoder = RecurrentEncoder(
        rnn_type=config['rnn_type'],
        embedding=src_embedding,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        bidirectional=config['bidirectional'],
        dropout=config['dropout']
    )
    bridge = Bridge(
        rnn_type=config['rnn_type'],
        hidden_size=config['hidden_size'],
        bidirectional=config['bidirectional']
    )
    rnn_cell = make_rnn_cell(config)
    attention = BilinearAttention(
        query_size=config['hidden_size'],
        key_size=config['hidden_size']
    )
    decoder = RecurrentDecoder(
        embedding=trg_embedding,
        rnn_cell=rnn_cell,
        attention=attention,
        hidden_size=config['hidden_size']
    )
    recurrent_seq2seq = RecurrentSeq2Seq(
        encoder=encoder,
        bridge=bridge,
        decoder=decoder
    )
    return recurrent_seq2seq

def make_conv_seq2seq(config):
    pass

def make_transformer(config):
    pass

def make_embedding(config):
    return nn.Embedding(
        num_embeddings=config['vocab_size'],
        embedding_dim=config['embed_size']
    )

def make_rnn_cell(config):
    if config['rnn_type'] == 'LSTM':
        return MultiLayerLSTMCells(
            input_size= 2 * config['trg_embedding']['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    elif config['rnn_type'] == 'GRU':
        return MultiLayerGRUCells(
            input_size= 2 * config['trg_embedding']['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    else:
        raise ValueError('No supporting.')