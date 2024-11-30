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
        if len(x.data.shape) < 2:
            raise ValueError(f"Expected input with shape (batch_size, {self.inputs}), got shape {x.data.shape}")
        if x.data.shape[-1] != self.inputs:
            raise ValueError(f"Expected input with {self.inputs} features, got {x.data.shape[-1]}")

        output = x.__matmul__(self.weight)
    
        if self.need_bias:
            output = output + self.bias

        if x.requires_grad or self.weight.requires_grad:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                if len(grad.shape) > 2:
                    weight_grad = sum(x_i.T.__matmul__(grad_i) 
                                   for x_i, grad_i in zip(x.data, grad))
                else:
                    weight_grad = x.data.T.__matmul__(grad)

                if self.weight.grad is None:
                    self.weight.grad = Tensor(np.zeros_like(self.weight.data))
                self.weight.grad.data += weight_grad

                if self.need_bias:
                    sum_axes = tuple(range(len(grad.shape)-1))
                    bias_grad = grad.sum(axis=sum_axes, keepdims=True)
                    if self.bias.grad is None:
                        self.bias.grad = Tensor(np.zeros_like(self.bias.data))
                    self.bias.grad.data += bias_grad

                return grad.__matmul__(self.weight.data.T)

            depends_on = [Dependency(x, grad_fn)]
        else:
            depends_on = []

        return Tensor(output.data, 
                     requires_grad=x.requires_grad or self.weight.requires_grad,
                     depends_on=depends_on)


    #Done
    def initialize(self):
        self.weight = Tensor(
            data=initializer((self.inputs, self.outputs), self.initialize_mode),
            requires_grad=True
        )
        
        assert self.weight.data.shape == (self.inputs, self.outputs), \
            f"Expected weight shape {(self.inputs, self.outputs)}, but got {self.weight.data.shape}"

        if self.need_bias:
            self.bias = Tensor(
                data=initializer((1, self.outputs), mode="zero"),
                requires_grad=True
            )
            assert self.bias.data.shape == (1, self.outputs), \
                f"Expected bias shape (1, {self.outputs}), but got {self.bias.data.shape}"
        
    #Done
    def zero_grad(self):
        if self.weight.grad is not None:
            self.weight.grad.data = np.zeros_like(self.weight.data, dtype=np.float64)
        
        if self.need_bias and self.bias.grad is not None:
            self.bias.grad.data = np.zeros_like(self.bias.data, dtype=np.float64)

    def parameters(self):
        params = [self.weight]
        if self.need_bias:
            params.append(self.bias)
        return params


    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs,
                                                                   self.outputs)
