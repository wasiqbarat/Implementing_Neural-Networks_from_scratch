import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from layer import Layer
from optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        for layer in self.layers:
            # Access the layer's parameters (weights, biases)
            for param in layer.parameters():
                if param.grad is not None:
                    # Update the parameter: param = param - learning_rate * param.grad
                    param.data = param.data - self.learning_rate * param.grad.data
    