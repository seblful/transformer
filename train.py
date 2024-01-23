from transformer import Transformer, WordEmbedder, TextPreprocessor

import os
import pandas as pd

HOME = os.getcwd()
DATA_PATH = os.path.join(HOME, 'data', 'data.csv')


def main():
    df = pd.read_csv(DATA_PATH)
    text_corpus = df['eng'].to_list()
    one_hot_text = TextPreprocessor(text_corpus=text_corpus,
                                    dict_size=5).one_hot_text

    print(one_hot_text.shape, one_hot_text)
    # w = WordEmbedder(text_corpus=xe)


if __name__ == "__main__":
    main()
