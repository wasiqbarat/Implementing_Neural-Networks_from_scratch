import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor import Tensor, Dependency
import numpy as np

# Done
def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    
    epsilon = Tensor(1e-15)
    
    clipped_preds = preds * (Tensor(1.0) - epsilon) + epsilon
    log_likelihood = label * clipped_preds.log()
    
    n_samples = Tensor(float(preds.shape[0]))
    n_samples_inv = n_samples ** (-1)  

    loss = -log_likelihood.sum() * n_samples_inv
    
    if preds.requires_grad or label.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return (clipped_preds - label) * n_samples_inv.data

        depends_on = [Dependency(preds, grad_fn)]
    else:
        depends_on = []

    return Tensor(loss.data, requires_grad=True, depends_on=depends_on)
