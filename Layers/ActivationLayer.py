from Layers import Layer
from FullyConnectedLayer import FullyConnectedLayer
import numpy as np


# Remember, the activation function isn't necessarily about deciding whether a neuron's output is passed to the next
# layer, but about transforming the neuron's output in a certain way (non-linearly).

class ActivationLayer(Layer):
    def __init__(self, activation_function, activation_derivative):
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation_function(input)
        return self.output

    # Backprop here applies the derivative of the activation function to the output error
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_derivative(self.input) * output_error


# These operations are performed element-wise for the tensors x:

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def step(x):
    return np.where(x > 0.5, 1, 0)


def tanH(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanH_derivative(x):
    return 1 - np.tanh(x) ** 2


def ReLU(x):
    return np.where(x > 0, x, 0)


def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)


def SoftMax(x):
    pass
