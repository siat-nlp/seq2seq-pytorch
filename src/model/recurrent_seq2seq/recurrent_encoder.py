from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.encoder import Encoder
from src.module.utils.constants import PAD_INDEX
from src.module.utils.sentence_clip import sentence_clip
from src.model.recurrent_seq2seq.recurrent_encoder_layer import RecurrentEncoderLayer

class RecurrentEncoder(Encoder):

    def __init__(self, rnn_type, embedding, hidden_size, num_layers=1, bidirectional=True, dropout=0.1):
        super(RecurrentEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.embedding = embedding
        embed_size = embedding.embedding_dim
        self.layers = nn.ModuleList([
            RecurrentEncoderLayer(
                rnn_type=rnn_type,
                input_size=embed_size,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                dropout=dropout
            )
        ])
        self.layers.extend(nn.ModuleList([
            RecurrentEncoderLayer(
                rnn_type=rnn_type,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                dropout=dropout
            ) for _ in range(num_layers)
        ]))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = dropout

    def forward(self, src):
        """
        :param src: LongTensor (batch_size, time_step)
        :return:
        """
        src = sentence_clip(src)
        src_embedding = self.embedding(src)
        src_embedding = F.dropout(src_embedding, p=self.dropout, training=self.training)
        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        src_lens, sort_index = src_lens.sort(descending=True)
        src_embedding = src_embedding.index_select(index=sort_index, dim=0)
        packed_src = pack_padded_sequence(src_embedding, src_lens, batch_first=True)
        final_states = []
        for layer in self.layers:
            packed_src, final_state = layer(packed_src)
            final_states.append(final_state)
        src, _ = pad_packed_sequence(packed_src, batch_first=True)
        src = self.layer_norm(src)
        return src, src_mask, final_states