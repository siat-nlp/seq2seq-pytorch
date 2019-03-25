from torch import nn
from src.module.encoder.transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from src.module.layer.feed_forward import FeedForward
from src.module.attention.scaled_dot_attention import ScaledDotAttention
from src.module.attention.multi_head_attention import MultiHeadAttention

def make_transformer(config):
    src_embedding = nn.Embedding(
        num_embeddings=config['src_embedding']['vocab_size'],
        embedding_dim=config['src_embedding']['embed_size']
    )
    trg_embedding = nn.Embedding(
        num_embeddings=config['trg_embedding']['vocab_size'],
        embedding_dim=config['trg_embedding']['embed_size']
    )