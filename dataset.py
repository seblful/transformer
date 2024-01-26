
from collections import Counter
import pandas as pd

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    def __init__(self,
                 csv_path,
                 dict_size,
                 sent_size):

        self.dict_size = dict_size
        self.sent_size = sent_size

        self.df = pd.read_csv(csv_path)  # .iloc[:1000]

        self.eng_texts_corpus = self.df['eng'].to_list()
        self.rus_texts_corpus = self.df['rus'].to_list()

        self.__eng_list_of_ind = None
        self.__rus_list_of_ind = None

    @property
    def eng_list_of_ind(self):
        if self.__eng_list_of_ind is None:
            self.__eng_list_of_ind = self.texts_to_index(self.eng_texts_corpus)

        return self.__eng_list_of_ind

    @property
    def rus_list_of_ind(self):
        if self.__rus_list_of_ind is None:
            self.__rus_list_of_ind = self.texts_to_index(self.rus_texts_corpus)

        return self.__rus_list_of_ind

    def texts_to_index(self,
                       texts_corpus):
        # Create list of all words and count it
        words = [word for sentence in texts_corpus for word in sentence.split()]
        word_counts = Counter(words)

        # Create a vocabulary from the tokens
        vocab = list(set(words))
        sorted_vocab = sorted(
            vocab, key=lambda x: word_counts[x], reverse=True)
        sorted_vocab = sorted_vocab[:self.dict_size - 2]

        # Create a dictionary to map words to indices
        word_to_idx = {word: i for i, word in enumerate(sorted_vocab, start=1)}

        # Convert the tokens to their corresponding indices
        indexed_data = [[word_to_idx.get(word, self.dict_size - 1)
                         for word in sentence.split()] for sentence in texts_corpus]

        indexed_tensors = [torch.tensor(
            sublist + [0] * (self.sent_size - len(sublist))) for sublist in indexed_data]

        return indexed_tensors

    def __len__(self):
        assert len(self.eng_texts_corpus) == len(self.rus_texts_corpus)
        return len(self.eng_texts_corpus)

    def __getitem__(self, index):

        eng_ind = self.eng_list_of_ind[index]
        rus_ind = self.rus_list_of_ind[index]

        return {"eng": eng_ind,
                "ru": rus_ind}
