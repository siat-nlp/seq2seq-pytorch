from trainer.make_model.make_recurrent_seq2seq import make_recurrent_seq2seq
from trainer.make_model.make_conv_seq2seq import make_conv_seq2seq
from trainer.make_model.make_transformer import make_transformer

def make_model(config):
    if config['model']['type'] == 'rnn':
        return make_recurrent_seq2seq(config)
    elif config['model']['type'] == 'cnn':
        return make_conv_seq2seq(config)
    elif config['model']['type'] == 'transformer':
        return make_transformer(config)
    else:
        raise ValueError('No supporting.')