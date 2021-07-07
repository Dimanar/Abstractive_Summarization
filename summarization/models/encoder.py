from torch import nn
from ..models.attentions import MultiHeadAttention
from ..models.secondary import AddNorm, PositionWiseFFN
from ..models.embeddings import WordEmbedding, PositionalEncoding


class EncoderLayer(nn.Module):

    """ Encoder block with """

    def __init__(self, device, d_model, ff_num_input, ff_num_hidden, ff_num_output, num_head, **kwargs):

        super(EncoderLayer, self).__init__(**kwargs)

        self.device = device
        self.d_model = d_model  # 256
        self.multi_head_attn = MultiHeadAttention(num_head, ff_num_input, ff_num_output)  # 8, 256, 256
        self.norm_attn = AddNorm(d_model)  # 256
        self.feed_forward = PositionWiseFFN(ff_num_input, ff_num_hidden, ff_num_output, device)  # 256, 128, 256
        self.feed_attn = AddNorm(d_model)  # 256
        self.attention_weights = None


    def forward(self, X, mask):

        # First residual connection and feed through multi head attention
        residual = X.clone()
        filtered, attn = self.multi_head_attn(X, X, X, mask)
        self.attention_weights = attn
        # Normalize batch
        filtered_norm = self.norm_attn(residual, filtered)

        # Second residual connection adn feed through FeedForwardNetwork
        residual = filtered_norm.clone()
        filtered = self.feed_forward(filtered_norm)
        # Normalize batch
        filtered_norm = self.feed_attn(residual, filtered)

        return filtered_norm.to(self.device)


class TransformerEncoder(nn.Module):

    def __init__(self, device, num_layers, d_model, vocab_size, ff_num_input,
                 ff_num_hidden, ff_num_output, num_head, **kwargs):

        super(TransformerEncoder, self).__init__(**kwargs)

        self.device = device
        self.d_model = d_model
        self.word_embedding = WordEmbedding(d_model, vocab_size)
        self.pos_embedding = PositionalEncoding(d_model)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(
                "block" + str(i),
                 EncoderLayer(device=device, d_model=d_model, ff_num_input=ff_num_input,
                              ff_num_hidden=ff_num_hidden, ff_num_output=ff_num_output,
                              num_head=num_head))


    def forward(self, input, mask):

        word_repr = self.word_embedding(input)
        word_pos_reps = self.pos_embedding(word_repr)

        self._attention_weights = [None] * len(self.blocks)

        X = word_pos_reps
        for i, blk in enumerate(self.blocks):
            X = blk(X, mask)
            # self._attention_weights[
            #     i] = blk.attention_weights

        return X.to(self.device)

