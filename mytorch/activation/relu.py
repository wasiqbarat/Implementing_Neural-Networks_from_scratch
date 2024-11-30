import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensor import Tensor, Dependency

def relu(x: Tensor) -> Tensor:
    data = np.maximum(0, x.data)
    requires_grad = x.requires_grad
    
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.where(x.data >= 0, 1.0, 0.0)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)
