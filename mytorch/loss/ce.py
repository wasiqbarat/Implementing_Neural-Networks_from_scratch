import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor import Tensor, Dependency
import numpy as np

# Done
def CategoricalCrossEntropy(preds: Tensor, label: Tensor) -> Tensor:
    """Compute Categorical Cross Entropy Loss.
    
    Args:
        preds: Predicted probabilities (after softmax), shape (batch_size, num_classes)
        label: One-hot encoded true labels, shape (batch_size, num_classes)
    
    Returns:
        Tensor containing the mean cross entropy loss
    """

    if len(preds.shape) != 2 or len(label.shape) != 2:
        raise ValueError(f"Expected 2D tensors, got shapes {preds.shape} and {label.shape}")
    if preds.shape != label.shape:
        raise ValueError(f"Shape mismatch: {preds.shape} vs {label.shape}")
    
    epsilon = 1e-15
    clipped_preds = np.clip(preds.data, epsilon, 1.0 - epsilon)
    
    batch_size = preds.shape[0]
    log_probs = np.log(clipped_preds)
    loss = -np.sum(label.data * log_probs) / batch_size
    
    if preds.requires_grad or label.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Gradient of cross entropy with respect to predictions
            # grad * (pred - label) / batch_size
            return grad * (clipped_preds - label.data) / batch_size

        depends_on = [Dependency(preds, grad_fn)]
    else:
        depends_on = []

    return Tensor(loss, 
                 requires_grad=preds.requires_grad or label.requires_grad,
                 depends_on=depends_on)


