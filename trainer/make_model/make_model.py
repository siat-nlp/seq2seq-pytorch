from trainer.make_model.make_recurrent_seq2seq import make_recurrent_seq2seq
from trainer.make_model.make_conv_seq2seq import make_conv_seq2seq
from trainer.make_model.make_transformer import make_transformer

def make_model(config):
    if config['type'] == 'rnn':
        return make_recurrent_seq2seq(config['rnn'])
    elif config['type'] == 'cnn':
        return make_conv_seq2seq(config['cnn'])
    elif config['type'] == 'transformer':
        return make_transformer(config['transformer'])
    else:
        raise ValueError('No supporting.')