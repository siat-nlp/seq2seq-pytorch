from torch import nn
from copy import deepcopy
from src.model.conv_seq2seq.conv_seq2seq import ConvSeq2Seq
from src.module.positional_embedding import PositionalEmbedding
from src.module.layer.conv_glu import ConvGLU
from src.module.layer.conv_relu import ConvReLU
from src.module.layer.feed_forward import FeedForward
from src.model.conv_seq2seq.conv_encoder import ConvEncoder
from src.model.conv_seq2seq.conv_decoder import ConvDecoder
from src.model.conv_seq2seq.conv_encoder_layer import ConvEncoderLayer
from src.model.conv_seq2seq.conv_decoder_layer import ConvDecoderLayer
from src.module.attention.dot_attention import DotAttention

def make_conv_seq2seq(config):
    src_embedding = nn.Embedding(
        num_embeddings=config['src_vocab_size'],
        embedding_dim=config['embed_size']
    )
    trg_embedding = nn.Embedding(
        num_embeddings=config['trg_vocab_size'],
        embedding_dim=config['embed_size']
    )
    positional_embedding = PositionalEmbedding(
        num_embeddings=config['num_positions'],
        embedding_dim=config['embed_size']
    )
    if config['activate'] == 'glu':
        encoder_conv = ConvGLU(
            input_size=config['hidden_size'],
            output_size=config['hidden_size'],
            kernel_size=config['kernel_size'],
            encode=True
        )
        decoder_conv = ConvGLU(
            input_size=config['hidden_size'],
            output_size=config['hidden_size'],
            kernel_size=config['kernel_size'],
            encode=False
        )
    elif config['activate'] == 'relu':
        encoder_conv = ConvReLU(
            input_size=config['hidden_size'],
            output_size=config['hidden_size'],
            kernel_size=config['kernel_size'],
            encode=True
        )
        decoder_conv = ConvReLU(
            input_size=config['hidden_size'],
            output_size=config['hidden_size'],
            kernel_size=config['kernel_size'],
            encode=False
        )
    feed_forward = FeedForward(
        input_size=config['hidden_size'],
        feed_forward_size=4 * config['hidden_size'],
        output_size=config['hidden_size']
    )
    conv_encoder_layer = ConvEncoderLayer(
        hidden_size=config['hidden_size'],
        conv = encoder_conv,
        feed_forward=deepcopy(feed_forward),
        dropout=config['dropout']
    )
    conv_encoder = ConvEncoder(
        embedding=src_embedding,
        positional_embedding=positional_embedding,
        layer=conv_encoder_layer,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    dot_attention = DotAttention(dropout=0)
    conv_decoder_layer = ConvDecoderLayer(
        hidden_size=config['hidden_size'],
        embed_size=config['embed_size'],
        conv=decoder_conv,
        attention=dot_attention,
        feed_forward=feed_forward,
        dropout=config['dropout']
    )
    conv_decoder = ConvDecoder(
        embedding=trg_embedding,
        positional_embedding=positional_embedding,
        layer=conv_decoder_layer,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    conv_seq2seq = ConvSeq2Seq(
        encoder=conv_encoder,
        decoder=conv_decoder
    )
    return conv_seq2seq