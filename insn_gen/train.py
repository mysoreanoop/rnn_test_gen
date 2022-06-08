import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
from collections import Counter

from model import LSTM

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_csv('data/data.csv')
        text = train_df['Data'].str.cat(sep=';')
        return text.split(';')

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        file = open('.dict', mode='w')
        file.write("{}\n".format(word_counts.keys()))
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - sequence_length

    def __getitem__(self, index):
        return (torch.tensor(self.words_indexes[index:index+sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+sequence_length+1]))


def train(dataset, model):
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(max_epochs):
        hidden, carry = model.init_state(sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (hidden, carry) = model(x, (hidden, carry))
            loss = criterion(y_pred.transpose(1, 2), y)

            hidden = hidden.detach()
            carry = carry.detach()

            loss.backward()
            optimizer.step()

            print(f"epoch {epoch}.{batch} | loss = {(loss.item()):3.3f}")


def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(';')
    hidden, carry = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (hidden, carry) = model(x, (hidden, carry))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words

max_epochs=10
batch_size=2048
sequence_length=4

dataset = Dataset()
model = LSTM(dataset)

train(dataset, model)
print(predict(dataset, model, text="or a5,a1,a2;addi a1,sp,8;mv t3,a0;"))
