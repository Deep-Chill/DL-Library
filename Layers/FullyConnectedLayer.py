from Layers import Layer
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


# # Remember, the activation function isn't necessarily about deciding whether a neuron's output is passed to the next
# # layer, but about transforming the neuron's output in a certain way (non-linearly).
#
# class ActivationLayer(Layer):
#     def __init__(self, activation_function, activation_derivative):
#         self.activation_function = activation_function
#         self.activation_derivative = activation_derivative
#
#     def forward_propagation(self, input):
#         self.input = input
#         self.output = self.activation_function(input)
#         return self.output
#
#     # Backprop here applies the derivative of the activation function to the output error
#     def backward_propagation(self, output_error, learning_rate):
#         return self.activation_derivative(self.input) * output_error

# class NNCompiler():
#     def __init__(self, input_size, n_layers, activation_function, activation_derivative, input_data, model_architecture,
#                  epochs, labels):
#         self.input_size = input_size
#         self.n_layers = n_layers
#         self.activation_function = activation_function
#         self.activation_derivative = activation_derivative
#         self.input_data = input_data
#         self.architecture = model_architecture
#         self.layer_type = self.architecture.layer_type
#         self.layer_size = self.architecture.layer_size
#         self.layer_output_size = self.architecture.layer_output_size
#         self.epochs = epochs
#         self.list_of_layers = []
#         for i in range(len(self.architecture)):
#             if self.layer_type[i] == 'FC':
#                 fc_layer = FullyConnectedLayer(self.layer_size[i], self.layer_output_size[i])
#                 self.list_of_layers.append(fc_layer)
#             elif self.layer_type[i] == 'AL':
#                 al_layer = ActivationLayer(self.activation_function, self.activation_derivative)
#                 self.list_of_layers.append(al_layer)
#         self.list_of_outputs = []
#         self.labels = labels
#
#         # if isinstance(self.list_of_layers[-1], FullyConnectedLayer):
#         #     if len(self.labels) != self.list_of_layers[-1].output:
#         #         raise ValueError("The size of labels does not match the size of output of the final layer in the nn")
#         # elif isinstance(self.list_of_layers[-1], ActivationLayer):
#         #     if len(self.labels) != self.list_of_layers[-2].output:
#         #         raise ValueError("The size of labels does not match the size of output of the final layer in the nn")
#
#
#     def first_layer(self):
#         self.list_of_outputs.append(self.list_of_layers[0].forward_propagation(self.input_data))
#
#     def forward_propagate(self):
#         for i, layers in enumerate(self.list_of_layers[1:]):
#             if isinstance(layers, FullyConnectedLayer):
#                 self.list_of_outputs.append(layers.forward_propagation(self.list_of_outputs[i-1]))
#             elif isinstance(layers, ActivationLayer):
#                 self.list_of_outputs.append(layers.forward_propagation(self.list_of_outputs[i-1]))
#
#
#
#
#     def backward_propagate(self):
#         for i in range(self.n_layers):
#             pass
#



# A model architecture class to define the order of layers in the model.
# The model architecture will be a list with dictionaries in the shape:
# [{'type': 'FC', 'layer_size': 10, 'output_size': 5}, {'type': 'AL', 'size': 10, 'output_size': 10, 'activation_function': 'sigmoid'}]
# Each item will be a layer. Each item has details about the layer.
# class ModelArchitecture():
#     def __init__(self, input):
#         self.architecture = input
#         self.layer_type = [i('type') for i in self.architecture]
#         self.layer_size = [i('layer_size') for i in self.architecture]
#         self.layer_output_size = [i('output_size') for i in self.architecture]
#












