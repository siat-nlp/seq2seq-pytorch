from src.module.seq2seq import Seq2Seq

class RecurrentSeq2Seq(Seq2Seq):

    def __init__(self, encoder, decoder):
        super(RecurrentSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        """
        src: LongTensor (batch_size, src_time_step)
        trg: LongTensor (batch_size, trg_time_step)
        """
        return self.decoder(self.encoder(src), trg)

    def decode(self, src, max_len):
        return self.decoder.greedy_decode(self.encoder(src), max_len)