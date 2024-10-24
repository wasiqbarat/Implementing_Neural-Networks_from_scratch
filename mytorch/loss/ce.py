from mytorch import Tensor

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    
    """
    Implements Categorical Cross Entropy loss: -Σ y_true * log(y_pred)
    
    Args:
        preds: Predicted probabilities tensor (output of softmax)
        label: One-hot encoded ground truth labels
        
    Returns:
        Tensor containing the categorical cross entropy loss
    """
    # Add small epsilon to prevent log(0)
    epsilon = Tensor(1e-15)
    
    # Clip predictions to prevent numerical instability
    # We add epsilon to avoid taking log of zero
    # We subtract epsilon from 1 to ensure sum of probabilities ≤ 1
    clipped_preds = preds * (Tensor(1.0) - epsilon) + epsilon
    
    # Calculate negative log likelihood
    # Note: we multiply by label to select only the predicted probability
    # of the correct class (since label is one-hot encoded)
    log_likelihood = label * clipped_preds.log()
    
    # Calculate mean negative log likelihood
    # Since we don't have sum with axis, we'll divide by number of samples
    n_samples = Tensor(float(preds.shape[0]))
    
    # Return negative mean (since we want to minimize -log likelihood)
    return -log_likelihood.sum() / n_samples

