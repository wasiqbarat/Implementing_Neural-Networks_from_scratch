import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor import Tensor
from layer import Layer
from dependency import Dependency

import numpy as np

class MaxPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), 
                                     (self.padding[0], self.padding[0]), 
                                     (self.padding[1], self.padding[1])), 
                            mode='constant', constant_values=-np.inf)
        else:
            x_padded = x.data
            
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.indices = np.zeros_like(output, dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = w * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        pool_region = x_padded[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.max(pool_region)
                        
                        idx = np.argmax(pool_region.reshape(-1))
                        self.indices[b, c, h, w] = idx
        
        out_tensor = Tensor(output, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                dx = np.zeros_like(x.data)
                
                for b in range(batch_size):
                    for c in range(channels):
                        for h in range(out_height):
                            for w in range(out_width):
                                h_start = h * self.stride[0]
                                h_end = h_start + self.kernel_size[0]
                                w_start = w * self.stride[1]
                                w_end = w_start + self.kernel_size[1]
                                
                                idx = self.indices[b, c, h, w]
                                h_idx = idx // self.kernel_size[1]
                                w_idx = idx % self.kernel_size[1]
                                
                                dx[b, c, h_start + h_idx, w_start + w_idx] += grad[b, c, h, w]
                
                return dx
            
            out_tensor.depends_on.append(Dependency(x, grad_fn))
            
        return out_tensor
    
    def __str__(self) -> str:
        return "max pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
