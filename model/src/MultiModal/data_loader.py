from sklearn.model_selection import train_test_split
import numpy as np
from dataclasses import dataclass
from torch.utils.data import random_split, DataLoader, Dataset


@dataclass
class DATALOADER:
    X: np.array
    Y: np.array
    tag: str = 'Regess'

    def __post_init__(self):
        if self.X.shape[0] < 5000:
            self.type = 'classical'
        else:
            self.type = 'DL'

    @staticmethod
    def train_datset_classic(**kwargs):
        x_train, x_test, y_train, y_test = train_test_split(**kwargs)
        return (x_train, x_test, y_train, y_test)

    @staticmethod
    def train_dataset_DL(**kwargs):
        dataset = kwargs.get('Dataset')
        total_len = dataset.shape[0]
        train_len = int(0.7 * total_len)
        val_len = int(0.15 * total_len)
        test_len = total_len - train_len - val_len
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_len, val_len, test_len])
        return (train_dataset, val_dataset, test_dataset)
