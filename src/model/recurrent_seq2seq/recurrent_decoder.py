import torch
from torch import nn
from src.module.decoder import Decoder
from src.module.utils.clone import clone

class RecurrentDecoder(Decoder):

    def __init__(self, embedding, attention, layer, num_layers=1, dropout=0.1, share_decoder_embedding=True):
        super(RecurrentDecoder, self).__init__()
        self.embedding = embedding
        self.layers = clone(layer, num_layers)
        self.attention = attention
        embed_size = embedding.embedding_dim
        vocab_size = embedding.num_embeddings
        hidden_size = layer.hidden_size
        self.output_projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embed_size)
        )
        self.generator = nn.Linear(embed_size, vocab_size)
        if share_decoder_embedding:
            self.generator.weight = embedding.weight
        self.dropout = dropout

    def forward(self, src, trg):
        src, src_mask, initial_states = src
        batch_size, max_len = trg.size()
        states = initial_states
        output = self.get_init_output(src, src_mask, initial_states)
        logit = []
        for i in range(max_len):
            token = trg[:, i:i+1]
            step_logit, states, output = self.step(src, src_mask, token, states, output)
            logit.append(step_logit)
        logit = torch.stack(logit, dim=1)
        return logit

    def step(self, src, src_mask, token, prev_states, prev_output):
        token_embedding = self.embedding(token).squeeze(1)
        state = torch.cat([token_embedding, prev_output], dim=1)
        states = []
        for i, layer in enumerate(self.layers):
            state = layer(state, prev_states[i])
            states.append(state)
        if isinstance(states[-1], tuple):   # LSTM
            top_hidden = states[-1][0]
        else:   # LSTM
            top_hidden = states[-1]
        context = self.attention(top_hidden, src, src, src_mask)
        output = self.output_projection(torch.cat([top_hidden, context], dim=1))
        logit = self.generator(output)
        return logit, states, output

    def get_init_output(self, src, src_mask, initial_states):
        if isinstance(initial_states[-1], tuple):  # LSTM
            init_top_hidden = initial_states[-1][0]
        else:   # GRU
            init_top_hidden = initial_states[-1]
        src_lens = src_mask.long().sum(dim=1, keepdim=True)
        src_mean = src.sum(dim=1, keepdim=False) / src_lens.float()
        init_output = self.output_projection(torch.cat([init_top_hidden, src_mean], dim=1))
        return init_output

class RecurrentDecoderLayer(nn.Module):

    def __init__(self):
        super(RecurrentDecoderLayer, self).__init__()

    def forward(self, *input):
        pass