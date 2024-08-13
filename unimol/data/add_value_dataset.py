"""
desc:	add one value to the labels
author:	Yangzhe Peng
date:	2024/01/15
"""

import torch
from functools import lru_cache
from unicore.data import UnicoreDataset

class AddValueDataset(UnicoreDataset):
    def __init__(self, labels, value):
        super().__init__()
        self.labels = labels
        self.value = value
    
    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.labels[index] + self.value

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return torch.tensor(samples)