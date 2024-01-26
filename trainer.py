import torch
import torch.nn as nn
import torch.optim as optim


class TransformerTrainer():
    def __init__(self,
                 transformer,
                 loader,
                 tgt_vocab_size,
                 model_save_path,
                 train_steps=100):

        self.transformer = transformer
        self.loader = loader

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(transformer.parameters(),
                                    lr=0.0001,
                                    betas=(0.9, 0.98),
                                    eps=1e-9)

        self.tgt_vocab_size = tgt_vocab_size
        self.train_steps = train_steps

        self.model_save_path = model_save_path

    def train(self):
        self.transformer.train()

        for epoch in range(self.train_steps):
            for data in self.loader:
                src_data, tgt_data = data['eng'], data['ru']
                self.optimizer.zero_grad()
                output = self.transformer(src_data, tgt_data[:, :-1])
                loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
                                      tgt_data[:, 1:].contiguous().view(-1))
                loss.backward()
                self.optimizer.step()
                print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    def save_model(self):
        torch.save(self.transformer.state_dict(), self.model_save_path)
