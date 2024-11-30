import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor import Tensor
from layer import Layer

import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), 
                                     (self.padding[0], self.padding[0]), 
                                     (self.padding[1], self.padding[1])), 
                            mode='constant')
        else:
            x_padded = x.data
            
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = w * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        receptive_field = x_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, c_out, h, w] = np.sum(receptive_field * self.weight.data[c_out])
                        
                        if self.need_bias:
                            output[b, c_out, h, w] += self.bias.data[c_out]
        
        out_tensor = Tensor(output, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                dx = np.zeros_like(x.data)
                dw = np.zeros_like(self.weight.data)
                if self.need_bias:
                    db = np.zeros(self.out_channels)
                
                return dx
            
            out_tensor.depends_on.append(Dependency(x, grad_fn))
            
        return out_tensor
    
    def initialize(self):
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        
        if self.initialize_mode == "xavier":
            limit = np.sqrt(6 / (fan_in + fan_out))
            weights = np.random.uniform(-limit, limit, 
                                     (self.out_channels, self.in_channels, *self.kernel_size))
        else:
            std = np.sqrt(2 / fan_in)
            weights = np.random.normal(0, std, 
                                    (self.out_channels, self.in_channels, *self.kernel_size))
            
        self.weight = Tensor(weights, requires_grad=True)
        
        if self.need_bias:
            self.bias = Tensor(np.zeros(self.out_channels), requires_grad=True)

    def zero_grad(self):
        if self.weight.grad is not None:
            self.weight.zero_grad()
        if self.need_bias and self.bias.grad is not None:
            self.bias.zero_grad()

    def parameters(self):
        if self.need_bias:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)
