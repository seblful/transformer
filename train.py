from dataset import TransformerDataset
from transformer import Transformer, WordEmbedder

import os
import pandas as pd

HOME = os.getcwd()
DATA_PATH = os.path.join(HOME, 'data', 'data.csv')


def main():
    transf_dataset = TransformerDataset(csv_path=DATA_PATH,
                                        dict_size=5120,
                                        sent_size=50)
    we = WordEmbedder()

    for text in transf_dataset:
        eng_text = text['eng_ind']
        print(we(eng_text).shape)
        break


if __name__ == "__main__":
    main()
