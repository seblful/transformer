import torch
import torch.nn as nn

import torch.nn.functional as F


class TextPreprocessor():
    def __init__(self,
                 text_corpus,
                 dict_size):
        self.text_corpus = text_corpus
        self.dict_size = dict_size

        self.__one_hot_text = None

    @property
    def one_hot_text(self):
        if self.__one_hot_text is None:
            self.__one_hot_text = self.text_to_one_hot(
                self.text_corpus)

        return self.__one_hot_text

    def text_to_one_hot(self,
                        text_corpus):
        # Create a vocabulary from the tokens
        vocab = set(
            word for sentence in text_corpus for word in sentence.split())
        print(len(vocab))

        # Create a dictionary to map words to indices
        word_to_idx = {word: i for i, word in enumerate(vocab)}

        # Convert the tokens to their corresponding indices
        indexed_data = [[word_to_idx[word]
                         for word in sentence.split()] for sentence in text_corpus]

        # Perform one-hot encoding using the one_hot function
        one_hot_encoded = F.one_hot(torch.tensor(
            indexed_data), num_classes=self.dict_size)

        return one_hot_encoded


class WordEmbedder(nn.Module):
    def __init__(self,
                 text_corpus):
        super().__init__()

        self.dict_size = 5120
        self.embedding_dim = 512

        self.text_preprocessor = TextPreprocessor(text_corpus=text_corpus,
                                                  dict_size=self.dict_size)

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
