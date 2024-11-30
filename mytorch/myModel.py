from tensor import Tensor, Dependency
from layer import Linear
from activation import relu, softmax
import model
import numpy as np

class MyModel(model.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = Linear(4, 32)
        self.linear2 = Linear(32, 16)
        self.linear3 = Linear(16, 3)

    def forward(self, x: Tensor):
        x = self.linear1.forward(x)
        x = relu(x)
        x = self.linear2.forward(x)
        x = relu(x)
        x = self.linear3.forward(x)
        x = softmax(x)
        return x


        
