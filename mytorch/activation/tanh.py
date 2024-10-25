import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensor import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """

    exp_x = x.exp()
    exp_neg_x = (-x).exp()
    
    numerator = exp_x - exp_neg_x
    denominator = exp_x + exp_neg_x
    
    return numerator / denominator