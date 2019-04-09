from torch import nn
from src.model.recurrent_seq2seq.recurrent_seq2seq import RecurrentSeq2Seq
from src.model.recurrent_seq2seq.recurrent_encoder import RecurrentEncoder
from src.model.recurrent_seq2seq.recurrent_decoder import RecurrentDecoder

def make_recurrent_seq2seq(config):
    src_embedding = nn.Embedding(
        num_embeddings=config['src_vocab_size'],
        embedding_dim=config['embed_size']
    )
    trg_embedding = nn.Embedding(
        num_embeddings=config['trg_vocab_size'],
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