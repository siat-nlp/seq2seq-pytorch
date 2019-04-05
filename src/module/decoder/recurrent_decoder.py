import torch
from torch import nn
from src.module.decoder.decoder import Decoder

class RecurrentDecoder(Decoder):

    def __init__(self, embedding, rnn_cell, attention, hidden_size):
        super(RecurrentDecoder, self).__init__()
        self.embedding = embedding
        self.rnn_cell = rnn_cell
        self.attention = attention
        self.output_projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embedding.embedding_dim)
        )

    def forward(self, src, trg):
        src, src_mask, initial_states = src
        batch_size, max_len = trg.size()
        states = initial_states
        output = self.get_init_output(src, src_mask, initial_states)
        logits = []
        for i in range(max_len):
            token = trg[:, i:i+1]
            logit, states, output = self.step(src, src_mask, token, states, output)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)
        return logits

    def step(self, src, src_mask, token, prev_states, prev_output):
        token_embedding = self.embedding(token).squeeze(1)
        rnn_input = torch.cat([token_embedding, prev_output], dim=1)
        states = self.rnn_cell(rnn_input, prev_states)
        if isinstance(states, tuple):
            top_hidden = states[0][-1]
        else:
            top_hidden = states[-1]
        context = self.attention(top_hidden, src, src, src_mask)
        output = self.output_projection(torch.cat([top_hidden, context], dim=1))
        logit = torch.mm(output, self.embedding.weight.t())
        return logit, states, output

    def get_init_output(self, src, src_mask, initial_states):
        if isinstance(initial_states, tuple):  # LSTM
            init_top_hidden = initial_states[0][-1]
        else:   # GRU
            init_top_hidden = initial_states[-1]
        src_lens = src_mask.long().sum(dim=1, keepdim=True)
        src_mean = src.sum(dim=1, keepdim=False) / src_lens.float()
        init_output = self.output_projection(torch.cat([init_top_hidden, src_mean], dim=1))
        return init_output