import torch
import torch.nn as nn

class Bridge(nn.Module):

    def __init__(self, rnn_type, hidden_size, bidirectional):
        super(Bridge, self).__init__()
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        encoder_output_size = hidden_size * (2 if bidirectional else 1)
        self.encoder_output_projection = nn.Linear(encoder_output_size, hidden_size)
        if rnn_type == 'LSTM':
            self.hidden_projection = nn.Linear(encoder_output_size, hidden_size)
            self.cell_projection = nn.Linear(encoder_output_size, hidden_size)
        elif rnn_type == 'GRU':
            self.hidden_projection = nn.Linear(encoder_output_size, hidden_size)
        else:
            raise ValueError('No Supporting.')

    def forward(self, encoder_output, final_encoder_states):
        # encoder_output: Tensor (batch_size, time_step, hidden_size * num_directions)
        # final_encoder_hidden: Tensor (num_layers * num_directions, batch_size, hidden_size)
        # final_encoder_cell: Tensor (num_layers * num_directions, batch_size, hidden_size)
        encoder_memory = self.encoder_output_projection(encoder_output)
        if self.rnn_type == 'LSTM':  # LSTM
            if self.bidirectional:
                final_encoder_states = (
                    torch.cat(
                        final_encoder_states[0].chunk(chunks=2, dim=0),
                        dim=2
                    ),
                    torch.cat(
                        final_encoder_states[1].chunk(chunks=2, dim=0),
                        dim=2
                    )
                )
            init_decoder_states = (
                torch.stack([
                    self.hidden_projection(hidden) for hidden in final_encoder_states[0]
                ], dim=0),
                torch.stack([
                    self.cell_projection(cell) for cell in final_encoder_states[1]
                ], dim=0)
            )
        else:  # GRU
            if self.bidirectional:
                final_encoder_states = torch.cat(
                    final_encoder_states.chunk(chunks=2, dim=0),
                    dim=2
                )
            init_decoder_states = torch.stack([
                self.hidden_projection(hidden) for hidden in final_encoder_states
            ], dim=0)
        return encoder_memory, init_decoder_states