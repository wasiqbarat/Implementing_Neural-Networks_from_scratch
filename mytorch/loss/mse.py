from mytorch import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    
    """
    Implements Mean Squared Error loss: MSE = (1/n) * Σ(y_pred - y_actual)²
    
    Args:
        preds: Predicted values tensor
        actual: Ground truth values tensor
        
    Returns:
        Tensor containing the mean squared error loss
    """
    # Calculate difference between predictions and actual values
    diff = preds - actual
    
    # Square the differences
    squared_diff = diff * diff
    
    # Calculate mean - since we don't have sum with axis support,
    # we'll use the fact that mean = sum/n where n is total elements
    n = Tensor(float(np.prod(preds.shape)))  # total number of elements
    
    # Calculate the mean of squared differences
    return squared_diff.sum() / n
