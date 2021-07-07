import torch
from torch import nn
import numpy as np


class WordEmbedding(nn.Module):

    """ Implement word embedding """

    def __init__(self, vector_dim, vocab_size, **kwargs):
        super(WordEmbedding, self).__init__(**kwargs)
        assert vector_dim > 1  # The dimension of each output vector must be more than 1

        self.vector_dim = vector_dim
        self.word_embeddings = nn.Embedding(vocab_size, vector_dim)

    def forward(self, words):
        """
        Get list of word and return the vectors representation from this words.
        Get list of string -> return list of vectors (or matrix)
        """
        return self.word_embeddings(words) * np.sqrt(self.vector_dim)


class PreTrainedEmbeddings(nn.Module):

    """ Implement pretrained word embedding (Glove or Word2Vec) """

    def __init__(self, embeddings, **kwargs):

        super(PreTrainedEmbeddings, self).__init__(**kwargs)
        self.word_embeddings = nn.Embedding(embeddings.size(0), embeddings.size(1),
                                            padding_idx=0)
        self.word_embeddings.weight = torch.nn.Parameter(embeddings)
        self.d_model = embeddings.size(1)

    def forward(self, x):
        """
        Get list of word and return the vectors representation from this words.
        Get list of string -> return list of vectors (or matrix). Here we use pretrained model
        for word representation.
        """
        return self.word_embeddings(x)


class PositionalEncoding(nn.Module):

    """ Implement simple PE function. """

    def __init__(self, dim, dropout=0.1, max_k=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_encoding', self._get_sin_cos_encoding(dim, max_k))

    def _get_angles(self, pos, i, d):
        """ get angle values """
        return pos * (1 / torch.pow(10000, (2 * (i // 2)) / d))

    def _get_sin_cos_encoding(self, d, i):
        """ Count positional vector """
        angles = self._get_angles(
            torch.arange(0, i).unsqueeze(1),
            torch.arange(0, d), d,
        )
        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])

        return torch.FloatTensor(angles).unsqueeze(0)

    def forward(self, word_repr):
        """
        Get matrix with word representation and add position information to every word.
        Position representation + word representation = word vectors with position information.
        """
        word_pos_repr = word_repr + self.pos_encoding[:, :word_repr.size(1)]
        return self.dropout(word_pos_repr)
