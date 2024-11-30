import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loss.ce import CategoricalCrossEntropy
from loss.mse import MeanSquaredError
