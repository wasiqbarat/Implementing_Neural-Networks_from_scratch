import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import numpy as np
from layer import Layer, Linear
from optimizer import Optimizer
from tensor import Tensor

class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate: float = 0.01):
        """Initialize SGD optimizer.
        Args:
            layers: List of layers to optimize
            learning_rate: Learning rate (default: 0.01)
        """
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        """Performs a single optimization step."""
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning rate: {self.learning_rate}")

        for layer in self.layers:
            for param in layer.parameters():
                if param is not None and param.requires_grad and param.grad is not None:
                    param.data -= self.learning_rate * param.grad.data

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        super().zero_grad()
