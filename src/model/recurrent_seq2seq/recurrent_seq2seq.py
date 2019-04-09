import torch
from src.module.utils import constants
from src.model.seq2seq import Seq2Seq

class RecurrentSeq2Seq(Seq2Seq):

    def __init__(self, encoder, bridge, decoder):
        super(RecurrentSeq2Seq, self).__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.decoder = decoder

    def forward(self, src, trg):
        """
        src: LongTensor (batch_size, src_time_step)
        trg: LongTensor (batch_size, trg_time_step)
        """
        encoder_output, final_encoder_states = self.encoder(src)
        src_memory, init_states = self.bridge(encoder_output, final_encoder_states)
        src_mask = (src != 0)
        max_len = src_mask.long().sum(dim=1, keepdim=False).max().item()
        src_mask = src_mask[:, 0:max_len]
        return self.decoder(src_memory, src_mask, init_states, trg)