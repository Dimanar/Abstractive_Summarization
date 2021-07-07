import torch
from torch import nn
from summarization.models import encoder, decoder
from summarization.models.secondary import Mask

"""
vocab_size, key_size, query_size, value_size, num_hiddens, 
norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
"""

class Transformer(nn.Module):

    def __init__(self, device, num_layers, d_model, vocab_size, ff_num_input,
                 ff_num_hidden, ff_num_output, num_head):

        super(Transformer, self).__init__()

        self.mask = Mask(device, 0)

        self.encoder = encoder.TransformerEncoder(
            device, num_layers, d_model, vocab_size, ff_num_input,
            ff_num_hidden, ff_num_output, num_head
        )

        self.decoder = decoder.TransformerDecoder(
            device, num_layers, d_model, vocab_size,
            ff_num_input, ff_num_hidden, ff_num_output,
            num_head
        )

        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.final_func = nn.Softmax()


    def forward(self, enc_inputs, dec_inputs):

        src_mask = self.mask.make_src_mask(enc_inputs)
        enc_outputs = self.encoder(enc_inputs, src_mask)

        trg_mask = self.mask.make_trg_mask(dec_inputs)
        out = self.decoder(dec_inputs, enc_outputs, trg_mask)

        return out
