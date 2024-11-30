import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor import Tensor, Dependency
from layer import Layer
from util import initializer

import numpy as np

class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode
        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        output = x.__matmul__(self.weight)
    
        if self.need_bias:
            output = output + self.bias

        if output.requires_grad:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                weight_grad = x.data.T.__matmul__(grad)
                bias_grad = grad.sum(axis=0) if self.need_bias else None
                self.weight.grad = weight_grad
                if self.need_bias:
                    self.bias.grad = bias_grad

                return grad.__matmul__(self.weight.data.T)

            depends_on = [Dependency(x, grad_fn)]
        else:
            epends_on = []

        return Tensor(output.data, requires_grad=True, depends_on=depends_on)


    #Done
    def initialize(self):
        "TODO: initialize weight by initializer function (mode)"
        
        self.weight = Tensor(
            data=initializer((self.inputs, self.outputs), self.initialize_mode),
            requires_grad=True
        )

        assert self.weight.data.shape == (self.inputs, self.outputs), f"Expected shape {(self.inputs, self.outputs)}, but got {self.weight.data.shape}"

        "TODO: initialize bias by initializer function (zero mode)"
        if self.need_bias:
            self.bias = Tensor(
                data=initializer((self.inputs, self.outputs), mode="zero"),
                requires_grad=True
            )
        
    #Done
    def zero_grad(self):
        "TODO: implement zero grad"
        if self.weight.grad is not None:
            self.weight.grad = np.zeros_like(self.weight, dtype=np.float64)  

    #Done
    def parameters(self):
        "TODO: return weights and bias"
        return [self.weight, self.bias]


    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs,
                                                                   self.outputs)

