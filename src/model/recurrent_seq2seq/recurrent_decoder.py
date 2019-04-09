import torch
from torch import nn
from src.module.decoder import Decoder
from src.model.recurrent_seq2seq.recurrent_decoder_layer import RecurrentDecoderLayer
from src.module.utils.clone import clone

class RecurrentDecoder(Decoder):

    def __init__(self, rnn_type, embedding, hidden_size, num_layers=1, dropout=0.1, share_decoder_embedding=True):
        super(RecurrentDecoder, self).__init__()
        self.rnn_type = rnn_type
        self.embedding = embedding
        embed_size = embedding.embedding_dim
        vocab_size = embedding.num_embeddings
        self.layers = nn.ModuleList([
            RecurrentDecoderLayer(
                rnn_type=rnn_type,
                input_size=2 * embed_size,
                hidden_size=hidden_size,
                dropout=dropout
            )
        ])
        self.layers.extend(nn.ModuleList([
            RecurrentDecoderLayer(
                rnn_type=rnn_type,
                hidden_size=hidden_size,
                dropout=dropout
            ) for _ in range(num_layers - 1)
        ]))
        self.layer_norm = nn.LayerNorm(hidden_size)
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
        hidden = torch.cat([token_embedding, prev_output], dim=1)
        states = []
        for i, layer in enumerate(self.layers):
            hidden, state = layer(hidden, prev_states[i])
            states.append(state)
        hidden = self.layer_norm(hidden)
        context = self.attention(hidden, src, src, src_mask)
        output = self.output_projection(torch.cat([hidden, context], dim=1))
        logit = self.generator(output)
        return logit, states, output

    def get_init_output(self, src, src_mask, initial_states):
        if self.rnn_type == 'LSTM':
            init_top_hidden = initial_states[-1][0]
        else:   # GRU
            init_top_hidden = initial_states[-1]
        src = src.masked_fill(src_mask.unsqueeze(-1), 0)
        src_lens = src_mask.long().sum(dim=1, keepdim=True)
        src_mean = src.sum(dim=1, keepdim=False) / src_lens.float()
        init_output = self.output_projection(torch.cat([init_top_hidden, src_mean], dim=1))
        return init_output