import Tensor
import numpy as np

# mean_squared_error used for regression(continuous values), output has a single value for each data point
def mean_squared_error(target, output):
    return np.mean((target-output)**2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# cross_entropy_loss used for classification
def cross_entropy_loss(target, output):
    pass