from torch import nn
from copy import deepcopy
from src.model.transformer.transformer import Transformer
from src.model.transformer.transformer_encoder import TransformerEncoder
from src.model.transformer.transformer_decoder import TransformerDecoder
from src.model.transformer.transformer_encoder_layer import TransformerEncoderLayer
from src.model.transformer.transformer_decoder_layer import TransformerDecoderLayer
from src.module.layer.feed_forward import FeedForward
from src.module.attention.scaled_dot_attention import ScaledDotAttention
from src.module.attention.multi_head_attention import MultiHeadAttention
from src.module.positional_embedding import PositionalEmbedding
import yaml
from data_process.utils import parse_path

def make_transformer(config):
    path = parse_path(config['data_process']['base_path'])
    data_log = yaml.load(open(path['log']['data_log']))
    share_src_trg_vocab = config['model']['share_src_trg_vocab']
    config = config['model'][config['model']['type']]
    if share_src_trg_vocab:
        src_embedding = nn.Embedding(
            num_embeddings=data_log['vocab_size'],
            embedding_dim=config['d_model']
        )
        trg_embedding = src_embedding
    else:
        src_embedding = nn.Embedding(
            num_embeddings=data_log['src_vocab_size'],
            embedding_dim=config['d_model']
        )
        trg_embedding = nn.Embedding(
            num_embeddings=data_log['trg_vocab_size'],
            embedding_dim=config['d_model']
        )
    positional_embedding = PositionalEmbedding(
        num_embeddings=config['num_positions'],
        embedding_dim=config['d_model'],
        learnable=False
    )
    scaled_dot_attention = ScaledDotAttention(dropout=0)
    multi_head_attention = MultiHeadAttention(
        attention=scaled_dot_attention,
        num_heads=config['num_heads'],
        hidden_size=config['d_model'],
        key_size=config['d_model'] // config['num_heads'],
        value_size=config['d_model'] // config['num_heads']
    )
    feed_forward = FeedForward(
        input_size=config['d_model'],
        feed_forward_size=4 * config['d_model'],
        output_size=config['d_model']
    )
    encoder_layer = TransformerEncoderLayer(
        hidden_size=config['d_model'],
        attention=deepcopy(multi_head_attention),
        feed_forward=deepcopy(feed_forward),
        dropout=config['dropout']
    )
    encoder = TransformerEncoder(
        embedding=src_embedding,
        positional_embedding=positional_embedding,
        layer=encoder_layer,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    decoder_layer = TransformerDecoderLayer(
        hidden_size=config['d_model'],
        self_attention=deepcopy(multi_head_attention),
        src_attention=deepcopy(multi_head_attention),
        feed_forward=deepcopy(feed_forward),
        dropout=config['dropout']
    )
    decoder = TransformerDecoder(
        embedding=trg_embedding,
        positional_embedding=positional_embedding,
        layer=decoder_layer,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder
    )
    return transformer