import numpy as np
from mytorch import Tensor, Dependency

def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Implements the Leaky ReLU activation function: 
    f(x) = x if x > 0
    f(x) = alpha * x if x <= 0
    
    Args:
        x: Input tensor
        alpha: Slope for negative values (default 0.01)
        
    Returns:
        Tensor with Leaky ReLU activation applied
    """
    # Leaky ReLU forward pass: max(alpha*x, x)
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
