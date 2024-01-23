import torch
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    def __init__(self,
                 df):
        self.df = df

        self.eng_text_corpus = df['eng'].to_list()
        self.rus_text_corpus = df['rus'].to_list()

    def __len__(self):
        assert len(self.eng_text_corpus) == len(self.rus_text_corpus)
        return len(self.eng_text_corpus)

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
