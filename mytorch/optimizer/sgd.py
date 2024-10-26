import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import numpy as np
from layer import Layer, Linear
from optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate
 
    def step(self):
        "Implement the SGD update rule"
        for layer in self.layers:
            for param in layer.weight:
                if param.requires_grad and param.grad is not None:                    
                    param.data -= self.learning_rate * param.data


