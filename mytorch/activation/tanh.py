import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    """
    Implements the hyperbolic tangent function: 
    tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with tanh activation applied
    """
    # We can break down tanh into steps using existing operations:
    # 1. Calculate e^x and e^(-x)
    # 2. Calculate numerator (e^x - e^(-x))
    # 3. Calculate denominator (e^x + e^(-x))
    # 4. Divide numerator by denominator
    
    exp_x = x.exp()
    exp_neg_x = (-x).exp()
    
    # Calculate numerator and denominator
    numerator = exp_x - exp_neg_x
    denominator = exp_x + exp_neg_x
    
    # Calculate final result
    return numerator / denominator