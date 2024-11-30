import sys
print(sys.executable)

import sys
import os
import pandas as pd
import torch
from tensor import Tensor
from torch.utils.data.dataset import Dataset
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pylab as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class CSVDataset(Dataset):
    def __init__(self, path: str):
        self.data = pd.get_dummies(pd.read_csv(path), columns=['Species']).astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        features = Tensor(row[:-3].values)
        label = Tensor([row.iloc[-3],row.iloc[-2],row.iloc[-1]])
        return features, label

    @property
    def classes(parameter_list):
        return ['Iris-setosa','Iris-versicolor','Iris-virginica']

path_train = 'Iris-Train.csv'
path_test = 'Iris-Test.csv'

dataset_train = CSVDataset(path_train)
dataset_test = CSVDataset(path_test)

from torch.utils.data import DataLoader
import numpy as np

def custom_collate_fn(batch):
    features = np.array([item[0].data for item in batch])
    labels = np.array([item[1].data for item in batch])
    return Tensor(features), Tensor(labels)

loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=5,
    shuffle=True,
    collate_fn=custom_collate_fn
)

loader_test = DataLoader(
    dataset=dataset_test,
    batch_size=5,
    shuffle=True,
    collate_fn=custom_collate_fn
)

from myModel import MyModel
from loss import CategoricalCrossEntropy
from optimizer import SGD

def one_epoch_learning(
    model: MyModel,
    criterion: CategoricalCrossEntropy,
    loader: DataLoader,
    optimizer: SGD,
) -> int:
    accs = 0
    for data, label in loader:
        optimizer.zero_grad()

        res: Tensor = mymodel(data)
        loss: Tensor = CategoricalCrossEntropy(res, label)
        loss.backward()

        optimizer.step()

        accs += (res.data.argmax(axis=0) == label.data.argmax(axis=0)).sum().item()

    return accs

def calculate_accuracy(
    model: MyModel, loader: DataLoader, criterion: CategoricalCrossEntropy
) -> int:
    accs = 0
    for data, label in loader:
        res: Tensor = model(data)
        accs += (res.data.argmax(axis=0) == label.data.argmax(axis=0)).sum().item()
    return accs

def train(
    model: MyModel,
    criterion: CategoricalCrossEntropy,
    loader_train: DataLoader,
    loader_test: DataLoader,
    optimizer: SGD,
    epoch: int,
):
    results_train = []
    results_test = []

    for i in tqdm(range(epoch)):
        res_train = one_epoch_learning(model, criterion, loader_train, optimizer)
        results_train.append(res_train / len(loader_train.dataset))
        res_test = calculate_accuracy(model, loader_test, criterion)
        results_test.append(res_test / len(loader_test.dataset))
    return results_train, results_test

if __name__ == "__main__":
    mymodel = MyModel()
    MyModel.summary(self=mymodel)
    
    print(f"batch   size\t= {loader_train.batch_size}")
    print(f"train   size\t= {len(dataset_train):,}")
    print(f"test    size\t= {len(dataset_test):,}")
    print(f"Class   names\t= {dataset_train.classes}")

    criterion = CategoricalCrossEntropy
    optimizer = SGD(MyModel.parameters(mymodel))
    train_accs, test_accs = train(mymodel, criterion, loader_train, loader_test, optimizer, 100)

    # Plot results
    plt.plot(train_accs, label="Train accuracy")
    plt.plot(test_accs, label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()