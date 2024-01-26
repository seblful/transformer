from dataset import TransformerDataset
from transformer import Transformer
from trainer import TransformerTrainer

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

HOME = os.getcwd()
DATA_PATH = os.path.join(HOME, 'data', 'data.csv')
MODEL_PATH = os.path.join(HOME, "checkpoints", "transformer.pth")

src_vocab_size = 5120
tgt_vocab_size = 5120
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 50
dropout = 0.1

batch_size = 64
train_steps = 100


def main():
    # Create dataset and loader
    transf_dataset = TransformerDataset(csv_path=DATA_PATH,
                                        dict_size=src_vocab_size,
                                        sent_size=max_seq_length)

    loader = DataLoader(transf_dataset, batch_size=batch_size, shuffle=True)

    # Create transformer and trainer
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                              num_heads, num_layers, d_ff, max_seq_length, dropout)

    trainer = TransformerTrainer(transformer=transformer,
                                 loader=loader,
                                 tgt_vocab_size=tgt_vocab_size,
                                 model_save_path=MODEL_PATH,
                                 train_steps=train_steps)

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    main()
