import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tensor import Tensor

def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten. 
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """
    data = ...
    req_grad = ...
    depends_on = ...
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
