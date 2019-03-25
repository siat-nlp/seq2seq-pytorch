from src.model.seq2seq import Seq2Seq

class Transformer(Seq2Seq):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        return self.decoder(self.encoder(src), trg)