import torch
from torch import nn
import numpy as np


class PositionWiseFFN(nn.Module):

    """ Implement FeedForward Position Encoding. Last layer without """

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 device, **kwargs):

        super(PositionWiseFFN, self).__init__(**kwargs)
        self.device = device
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        """ Simple feed forward network. """
        return self.dense2(self.relu(self.dense1(X))).to(self.device)


class AddNorm(nn.Module):

    """ Residual connection and normalization:  alpha * (x_i - mean) / (std - eps) + bias """

    def __init__(self, features, eps=1e-6, **kwargs):

        super(AddNorm, self).__init__(**kwargs)

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, residual, prev_layer):

        X = residual + prev_layer

        mean = torch.mean(X, axis=1, keepdim=True)
        std = torch.std(X, axis=1, keepdim=True)

        return self.alpha * (X - mean) / (std + self.eps) + self.bias


class Mask(object):

    def __init__(self, device, pad):
        self.src_pad_idx = pad
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

