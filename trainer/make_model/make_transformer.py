from torch import nn
import copy.deepcopy as copy
from src.model.transformer import Transformer
from src.module.encoder.transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from src.module.decoder.transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from src.module.layer.feed_forward import FeedForward
from src.module.attention.scaled_dot_attention import ScaledDotAttention
from src.module.attention.multi_head_attention import MultiHeadAttention
from src.module.embedding.positional_embedding import PositionalEmbedding

def make_transformer(config):
    src_embedding = nn.Embedding(
        num_embeddings=config['src_vocab_size'],
        embedding_dim=config['d_model']
    )
    src_positional_embedding = PositionalEmbedding(
        num_embeddings=config['src_vocab_size'],
        embedding_dim=config['d_model'],
        learnable=False
    )
    trg_embedding = nn.Embedding(
        num_embeddings=config['trg_vocab_size'],
        embedding_dim=config['d_model']
    )
    trg_positional_embedding = PositionalEmbedding(
        num_embeddings=config['trg_vocab_size'],
        embedding_dim=config['d_model'],
        learnable=False
    )
    scaled_dot_attention = ScaledDotAttention(dropout=0)
    multi_head_attention = MultiHeadAttention(
        attention=scaled_dot_attention,
        num_heads=config['num_heads'],
        hidden_size=config['d_model'],
        key_size=config['d_model'] / config['num_heads'],
        value_size=config['d_model'] / config['num_heads']
    )
    feed_forward = FeedForward(
        hidden_size=config['d_model'],
        feed_forward_size=2 * config['d_model']
    )
    encoder_layer = TransformerEncoderLayer(
        hidden_size=config['d_model'],
        attention=copy(multi_head_attention),
        feed_forward=copy(feed_forward),
        dropout=config['dropout']
    )
    encoder = TransformerEncoder(
        embedding=src_embedding,
        positional_embedding=src_positional_embedding,
        layer=encoder_layer,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    decoder_layer = TransformerDecoderLayer(
        hidden_size=config['d_model'],
        self_attention=copy(multi_head_attention),
        src_attention=copy(multi_head_attention),
        feed_forward=copy(feed_forward),
        dropout=config['dropout']
    )
    decoder = TransformerDecoder(
        embedding=trg_embedding,
        positional_embedding=trg_positional_embedding,
        layer=decoder_layer,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder
    )
    return transformer