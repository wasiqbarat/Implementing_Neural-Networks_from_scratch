import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import os
from torch.utils.data import DataLoader

class CSVDataset(Dataset):
    def __init__(self, path: str):
        self.data = pd.get_dummies(pd.read_csv(path), columns=['Species']).astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        features = torch.tensor(row[:-3].values, dtype=torch.float32)
        label = torch.tensor([row.iloc[-3],row.iloc[-2],row.iloc[-1]], dtype=torch.float32)
        return features, label

    @property
    def classes(parameter_list):
        return ['Iris-setosa','Iris-versicolor','Iris-virginica']
    
    base_path = os.getcwd()


base_path = os.getcwd()
#TODO change pathes
path_train = os.path.join(base_path, "Iris-Train.csv")
path_test = os.path.join(base_path, "Iris-Test.csv")

dataset_train = CSVDataset(path_train)
dataset_test = CSVDataset(path_test)



loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=10,
    shuffle=True
)

loader_test = DataLoader(
    dataset=dataset_test,
    batch_size=10,
    shuffle=True
)

