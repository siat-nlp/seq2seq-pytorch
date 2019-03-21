import torch
from src.module.utils import constants
from src.model.seq2seq import Seq2Seq

class RecurrentSeq2Seq(Seq2Seq):

    def __init__(self, encoder, bridge, decoder):
        super(RecurrentSeq2Seq, self).__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.decoder = decoder

    def forward(self, src, trg):
        """
        src: LongTensor (batch_size, src_time_step)
        trg: LongTensor (batch_size, trg_time_step)
        """
        encoder_output, final_encoder_states = self.encoder(src)
        src_memory, init_states = self.bridge(encoder_output, final_encoder_states)
        src_mask = (src != 0)
        max_len = src_mask.long().sum(dim=1, keepdim=False).max().item()
        src_mask = src_mask[:, 0:max_len]
        return self.decoder(src_memory, src_mask, init_states, trg)

    def decode(self, src, max_len):
        encoder_output, final_encoder_states = self.encoder(src)
        src_memory, init_states = self.bridge(encoder_output, final_encoder_states)
        src_mask = (src != 0)
        init_output = self.decoder.get_init_output(src_memory, src_mask, init_states)
        batch_size = src_memory.size(0)
        token = torch.tensor([constants.SOS_INDEX] * batch_size).unsqueeze(1).cuda()
        states = init_states
        output = init_output
        outputs = []
        for _ in range(max_len):
            logit, states, output = self._decoder.step(src_memory, src_mask, token, states, output)
            token = torch.max(logit, dim=1, keepdim=True)[1]
            outputs.append(token[:, 0])
        outputs = torch.stack(outputs, dim=1)
        return outputs

    # def beam_decode(self, src, src_lens, max_len, beam_size):
    #     # src: Tensor (batch_size, time_step)
    #     # src_lens: list (batch_size,)
    #     # max_len: int
    #     # beam_size: int
    #     src_memory, src_mask, init_states = self._encode(src, src_lens)
    #     init_output = self._decoder.get_init_output(src_memory, src_lens, init_states)
    #     batch_size, time_step, hidden_size = src_memory.size()
    #     src_memory = src_memory.repeat(beam_size, 1, 1, 1).view(beam_size * batch_size, time_step,
    #                                                             hidden_size).contiguous()
    #     src_mask = src_mask.repeat(beam_size, 1, 1).view(beam_size * batch_size, time_step).contiguous()
    #     beamer = Beamer(
    #         states=init_states,
    #         output=init_output,
    #         beam_size=beam_size,
    #         remove_repeat_triple_grams=True
    #     )
    #     for _ in range(max_len):
    #         token, states, output = beamer.pack_batch()
    #         logit, states, output = self._decoder.step(
    #             src_memory, src_mask, token, states, output
    #         )
    #         log_prob = F.log_softmax(logit, dim=-1)
    #         log_prob, token = log_prob.topk(k=beam_size, dim=-1)
    #         beamer.update_beam(token, log_prob, states, output)
    #     outputs = beamer.get_best_sequences(max_len)
    #     return outputs