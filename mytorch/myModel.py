from tensor import Tensor, Dependency
from layer import Linear
from activation import relu, softmax
import model
import numpy as np

class MyModel(model.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = Linear(4, 5)
        self.linear2 = Linear(5, 3)

    def forward(self, x: Tensor):
        x = self.linear1(x)
        x = relu(x)
        print(x)
        print(x.shape)
        x = self.linear2(x)
        x = softmax(x)
        print(x)
        return x
