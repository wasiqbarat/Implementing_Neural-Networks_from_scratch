import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    
    diff = preds - actual
    
    squared_diff = diff * diff
    
    n = Tensor(float(np.prod(preds.shape))) 
    
    return squared_diff.sum() / n
