import numpy as np
from mytorch import Tensor, Dependency


def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """

    exp_x = x.exp()
    # Create a matrix of ones with shape (num_classes, 1)
    # This will help us sum along the classes axis through matrix multiplication
    ones = Tensor(np.ones((x.shape[1], 1)))
    
    # Calculate sum of exponentials for each sample
    # exp_x @ ones will give us shape (batch_size, 1)
    sum_exp = exp_x @ ones
    
    # Now we need to divide exp(x) by sum_exp
    # Using the hint that a/b = a * (b^-1)
    # We can use broadcasting to divide each row by its sum
    return exp_x * (Tensor(1.0) / sum_exp)
