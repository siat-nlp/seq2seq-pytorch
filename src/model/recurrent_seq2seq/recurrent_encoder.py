from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.encoder import Encoder
from src.module.utils.constants import PAD_INDEX
from src.module.utils.clone import clone

class RecurrentEncoder(Encoder):

    def __init__(self, embedding, layer, num_layers=1, dropout=0.1):
        super(RecurrentEncoder, self).__init__()
        self.embedding = embedding
        self.dropout = dropout
        self.layers = clone(layer, num_layers)
        hidden_size = layer.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = dropout

    def forward(self, src):
        """
        :param src: LongTensor (batch_size, time_step)
        :return:
        """
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