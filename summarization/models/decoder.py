from torch import nn
from ..models.attentions import MultiHeadAttention
from ..models.secondary import AddNorm, PositionWiseFFN
from ..models.embeddings import WordEmbedding, PositionalEncoding


class DecoderLayer(nn.Module):

    """
    Encoder block with:
      - MultiHeadAttention
      - AddNorm
      - FeedForwardNetwork
      - AddNorm
     """

    def __init__(self, device, d_model, ff_num_input, ff_num_hidden, ff_num_output, num_head, **kwargs):

        super(DecoderLayer, self).__init__(**kwargs)

        self.device = device
        self.d_model = d_model  # 256
        self.multi_head_attn_first = MultiHeadAttention(num_head, ff_num_input, ff_num_output)  # 8, 256, 256
        self.multi_head_attn_second = MultiHeadAttention(num_head, ff_num_input, ff_num_output)  # 8, 256, 256
        self.norm_attn_first = AddNorm(d_model)  # 256
        self.norm_attn_second = AddNorm(d_model)  # 256
        self.feed_forward = PositionWiseFFN(ff_num_input, ff_num_hidden, ff_num_output, device)  # 256, 128, 256
        self.feed_norm = AddNorm(d_model)  # 256
        self.attention_weights = [None, None]


    def forward(self, X, encoder, mask):

        # First residual connection and feed through  masked multi head attention
        residual = X.clone()
        filtered, dec_attn = self.multi_head_attn_first(X, X, X, mask=mask)
        self.attention_weights[0] = dec_attn
        # Batch normalize
        filtered_norm = self.norm_attn_first(residual, filtered)

        # Second residual connection adn feed through multi head attention
        residual = filtered_norm.clone()
        filtered, enc_attn = self.multi_head_attn_second(encoder, encoder, filtered_norm)
        self.attention_weights[1] = enc_attn
        # Batch normalize
        filtered_norm = self.norm_attn_second(residual, filtered)

        # Third residual connection adn feed through FeedForwardNetwork
        residual = filtered_norm.clone()
        result = self.feed_forward(filtered_norm)
        # Batch normalize
        result = self.feed_norm(residual, result)

        return result.to(self.device)


class TransformerDecoder(nn.Module):

    def __init__(self, device, num_layers, d_model, vocab_size, ff_num_input,
                 ff_num_hidden, ff_num_output, num_head, **kwargs):

        super(TransformerDecoder, self).__init__(**kwargs)

        self.device = device
        self.d_model = d_model
        self.word_embedding = WordEmbedding(d_model, vocab_size)
        self.pos_embedding = PositionalEncoding(d_model)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(
                "block" + str(i),
                 DecoderLayer(device, d_model, ff_num_input, ff_num_hidden, ff_num_output, num_head))


    def forward(self, input_dec, output_enc, mask):

        word_repr = self.word_embedding(input_dec)
        word_pos_reps = self.pos_embedding(word_repr)

        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]

        X = word_pos_reps
        for i, blk in enumerate(self.blocks):
            X = blk(X, output_enc, mask)

            # self._attention_weights[0][
            #     i] = blk.attention1.attention.attention_weights[0]
            #
            # self._attention_weights[1][
            #     i] = blk.attention2.attention.attention_weights[1]

        return X.to(self.device)
