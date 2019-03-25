from src.model.seq2seq import Seq2Seq

class ConvSeq2Seq(Seq2Seq):

    def __init__(self, encoder, decoder):
        super(ConvSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        return self.decoder(self.encoder(src), trg)

    def greedy_decode(self, src):
        pass

    def beam_decode(self, src):
        pass