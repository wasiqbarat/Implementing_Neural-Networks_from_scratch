import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensor import Tensor, Dependency

def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """
    TODO: implement leaky_relu function.
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn
    hint: use np.where like Relu method but for LeakyRelu
    """

    data = np.where(x.data > 0, x.data, alpha * x.data)
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            # Leaky ReLU gradient: 1 if x > 0, alpha if x <= 0
            return np.where(x.data > 0, grad, alpha * grad)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
