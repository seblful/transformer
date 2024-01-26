from dataset import TransformerDataset
from transformer import Transformer

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

HOME = os.getcwd()
DATA_PATH = os.path.join(HOME, 'data', 'data.csv')


def main():
    transf_dataset = TransformerDataset(csv_path=DATA_PATH,
                                        dict_size=5120,
                                        sent_size=50)
    loader = DataLoader(transf_dataset, batch_size=64, shuffle=True)

    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                              num_heads, num_layers, d_ff, max_seq_length, dropout)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(),
                           lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(100):
        for data in loader:
            src_data, tgt_data = data['eng'], data['ru']
            optimizer.zero_grad()
            output = transformer(src_data, tgt_data[:, :-1])
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                             tgt_data[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


if __name__ == "__main__":
    main()
