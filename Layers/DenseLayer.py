from .Layers import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # the "input error" is a measure of "how wrong" each neuron in the input layer was during the forward pass
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


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

class NNCompiler():
    def __init__(self, input_size, n_layers, activation_function, activation_function_derivative):
        self.input_size = input_size
        self.n_layers = n_layers
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative














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
    return 1 - np.tanh(x)**2

def ReLU(x):
    return np.where(x > 0, x, 0)

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def SoftMax(x):
    pass










