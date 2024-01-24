import torch
import torch.nn as nn

import torch.nn.functional as F


class WordEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dict_size = 5120
        self.embedding_dim = 512

        self.embedder = nn.Embedding(num_embeddings=self.dict_size,
                                     embedding_dim=self.embedding_dim)

    def forward(self, x):
        x = self.embedder(x)

        return x


class PosEncoder():
    pass


class Attention():
    pass


class BaseMultiHeadAttention():
    pass


class SimpleMultiHeadAttention():
    pass


class MaskedMultiHeadAttention():
    pass


class TransformerEncoder():
    pass


class TransformerDecoder():
    pass


class Transformer():
    pass
