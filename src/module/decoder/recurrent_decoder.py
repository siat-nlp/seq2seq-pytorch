import torch
import torch.nn as nn
from src.module.decoder.decoder import Decoder

class RecurrentDecoder(Decoder):

    def __init__(self, embedding, rnn_cell, attention, hidden_size):
        super(RecurrentDecoder, self).__init__()
        self.embedding = embedding
        self.rnn_cell = rnn_cell
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.attn = attention
        self.output_projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embedding.embedding_dim)
        )

    def forward(self, src_memory, src_mask, init_states, trg):
        batch_size, max_len = trg.size()
        states = init_states
        output = self.get_init_output(src_memory, src_mask, init_states)
        logits = []
        for i in range(max_len):
            token = trg[:, i: i + 1]
            logit, states, output = self.step(src_memory, src_mask, token, states, output)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)
        return logits

    def step(self, src_memory, src_mask, token, prev_states, prev_output):
        # src_memory: Tensor (batch_size, time_step, hidden_size)
        # src_mask: Tensor (batch_size, time_step)
        # token: Tensor (batch_size, 1)
        # prev_states: tuple (hidden, cell)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        # prev_output: (batch_size, embed_size)
        token_embedding = self.embedding(token).squeeze(1)
        rnn_input = torch.cat([token_embedding, prev_output], dim=1)
        states = self.rnn_cell(rnn_input, prev_states)
        if isinstance(states, tuple):   # LSTM
            top_hidden = states[0][-1]
        else:   # GRU
            top_hidden = states[-1]
        query = self.query_projection(top_hidden)
        context = self.attn(query, src_memory, src_memory, src_mask)
        output = self.output_projection(torch.cat([top_hidden, context], dim=1))
        logit = torch.mm(output, self.embedding.weight.t())
        return logit, states, output

    def get_init_output(self, src_memory, src_mask, init_states):
        if isinstance(init_states, tuple):  # LSTM
            init_top_hidden = init_states[0][-1]
        else:   # GRU
            init_top_hidden = init_states[-1]
        src_lens = src_mask.long().sum(dim=1, keepdim=True)
        src_mean = src_memory.sum(dim=1, keepdim=False) / src_lens.float()
        init_output = self.output_projection(torch.cat([init_top_hidden, src_mean], dim=1))
        return init_output