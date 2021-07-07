import torch
from torch import nn
import numpy as np
from ..preparation import clone


class Attention(object):

    """ Compute scale dot product attention """

    def __call__(self, query, key, value, mask=None, dropout=None):
        """
        Get 3 matrices with word representation and return ONE matrix.
        Here we find similarity between filter(query @ key) and value.
        """
        d_k = query.size(-1)
        # Multiply query and key to get filter
        attn_filer = torch.matmul(query, key.transpose(-2, -1))

        # Scale result
        attn_filer = attn_filer / np.sqrt(d_k)

        # Mask use in decoder
        if mask is not None:
            attn_filer = attn_filer.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_filer = torch.softmax(attn_filer, dim=-1)

        if dropout is not None:
            attn_filer = dropout(attn_filer)

        # Matmul with Value to get final result
        score = torch.matmul(attn_filer, value)

        return score, attn_filer


class MultiHeadAttention(nn.Module):

    """ Compute multiHead self-attention """

    def __init__(self, num_head, ff_num_input_output, ff_num_hidden, dropout=0.1, **kwargs):

        super(MultiHeadAttention, self).__init__(**kwargs)

        self.d_model, self.dd_hidden = ff_num_input_output, ff_num_hidden
        self.num_head = num_head
        self.dropout = nn.Dropout(p=dropout)
        self.W_Q = nn.Linear(self.d_model, self.dd_hidden * self.num_head)
        self.W_K = nn.Linear(self.d_model, self.dd_hidden * self.num_head)
        self.W_V = nn.Linear(self.d_model, self.dd_hidden * self.num_head)

        self.LinearFinal = nn.Linear(self.dd_hidden * self.num_head, self.d_model)
        self.attention = Attention()

    # def __getattr__(self, attr):
    #     return self.LinearFinal.weight


    def forward(self, query, key, value, mask=None):
        """
        Here we use Attention for getting MULTI-HEAD attention.
        After getting result from all Attention part we concatenate result and feed through FFD.
        FFD helps to change shape from concatenation matrix.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        residual, batch_size = query, query.size(0)

        # 1) Do all the linear projections in batch from d_model => (h x d_k)
        q = self.W_Q(query).view(batch_size, -1, self.num_head, self.dd_hidden).transpose(1, 2)
        k = self.W_K(key).view(batch_size, -1, self.num_head, self.dd_hidden).transpose(1, 2)
        v = self.W_V(value).view(batch_size, -1, self.num_head, self.dd_hidden).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attention = self.attention(q, k, v, mask=mask,
                                      dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.dd_hidden)

        # 4) Apply final linear module for output
        output = self.LinearFinal(x)

        return output, attention