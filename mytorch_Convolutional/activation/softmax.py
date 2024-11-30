import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensor import Tensor, Dependency

def softmax(tensor: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """
    
    max_vals = Tensor(tensor.data.max(axis=-1, keepdims=True))
    shifted = tensor - max_vals
    
    exp_vals = shifted.exp()
    
    ones = Tensor(np.ones((tensor.shape[-1], 1)))
    
    sum_exp = exp_vals @ ones

    sum_inv = sum_exp ** -1
    
    return exp_vals * sum_inv
