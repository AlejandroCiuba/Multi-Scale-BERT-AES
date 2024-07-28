# ASAP Dataset Dataset
# Created by Alejandro Ciuba, alc307@pitt.edu
from torch.utils.data import Dataset

import torch

import numpy as np
import pandas as pd


class ToEncoded():

    def __init__(self):
        pass


class ASAPDataset(Dataset):

    X_train, X_test = None, None
    X_valid, y_valid = None, None
    y_train, y_test = None, None

    def __init__(self, data: pd.DataFrame, train_splits, valid_split, 
                 test_split, prompt: int):
        
        super().__init__()

        train = data[(data.split.isin(train_splits)) & (data.essay_set == prompt)]
        valid = data[(data.split == valid_split) & (data.essay_set == prompt)]
        test = data[(data.split == test_split) & (data.essay_set == prompt)]

        self.X_train, self.y_train = train.essay, torch.from_numpy(train.score.to_numpy())
        self.X_valid, self.y_valid = valid.essay, torch.from_numpy(valid.score.to_numpy())
        self.X_test, self.y_test = test.essay, torch.from_numpy(test.score.to_numpy())

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def get_valid(self):
        return self.X_valid, self.y_valid

    def get_test(self):
        return self.X_test, self.y_test
