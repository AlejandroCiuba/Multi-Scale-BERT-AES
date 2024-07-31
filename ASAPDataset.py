# ASAP Dataset Dataset
# Created by Alejandro Ciuba, alc307@pitt.edu
from encoder import encode_documents
from torch.utils.data import Dataset

import torch

import numpy as np
import pandas as pd


class ToEncoded():

    tokenizer = None
    chunk_sizes = None

    def __init__(self, tokenizer, chunk_sizes):

        self.tokenizer = tokenizer
        self.chunk_sizes = chunk_sizes

    def __call__(self, sample):

        X, y = sample
        print(y)

        X_word_level, _ = encode_documents([X], self.tokenizer, max_input_length=512)
        X_reps = [X_word_level]

        for chunk_size in self.chunk_sizes:

            doc_rep, _ = encode_documents([X], self.tokenizer, max_input_length=chunk_size)
            X_reps.append(doc_rep)

        return X_reps, y


class ASAPDataset(Dataset):

    X_train, X_test = None, None
    X_valid, y_valid = None, None
    y_train, y_test = None, None

    transform = None

    def __init__(self, data: pd.DataFrame, train_splits, valid_split, 
                 test_split, prompt: int, transform = None):
        
        super().__init__()

        if transform is not None:
            self.transform = transform

        train = data[(data.split.isin(train_splits)) & (data.essay_set == prompt)]
        valid = data[(data.split == valid_split) & (data.essay_set == prompt)]
        test = data[(data.split == test_split) & (data.essay_set == prompt)]

        self.X_train, self.y_train = train.essay.to_list(), torch.from_numpy(train.score.to_numpy())
        self.X_valid, self.y_valid = valid.essay.to_list(), torch.from_numpy(valid.score.to_numpy())
        self.X_test, self.y_test = test.essay.to_list(), torch.from_numpy(test.score.to_numpy())

    def __getitem__(self, index: int):

        if self.transform is not None:
            return self.transform((self.X_train[index], self.y_train[index]))

        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return len(self.X_train)

    def get_valid(self):
        return self.X_valid, self.y_valid

    def get_test(self):
        return self.X_test, self.y_test
