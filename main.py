from Tensor.Tensors import Tensor
from Layers import ActivationLayer, FullyConnectedLayer, Layers
from LossFunctions.Loss import mean_squared_error, mse_derivative

import numpy as np
sigmoid = ActivationLayer.sigmoid
sigmoid_derivative = ActivationLayer.sigmoid_derivative

ActivationLayer = ActivationLayer.ActivationLayer
FullyConnectedLayer = FullyConnectedLayer.FullyConnectedLayer
Layer = Layers.Layer


class NNCompiler():
    def __init__(self, input_size, n_layers, activation_function, activation_derivative, input_data, model_architecture,
                 epochs, target, learning_rate):
        self.input_size = input_size
        self.n_layers = n_layers
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.input_data = input_data
        self.architecture = model_architecture
        # print(type(self.architecture))
        # print(self.architecture)
        # print(self.architecture.layer_type)
        self.layer_type = model_architecture.layer_type
        self.layer_size = self.architecture.layer_size
        self.layer_output_size = self.architecture.layer_output_size
        self.epochs = epochs
        self.list_of_layers = []
        for i in range(len(self.layer_type)):
            if self.layer_type[i] == 'FC':
                fc_layer = FullyConnectedLayer(self.layer_size[i], self.layer_output_size[i])
                self.list_of_layers.append(fc_layer)
            elif self.layer_type[i] == 'AL':
                al_layer = ActivationLayer(self.activation_function, self.activation_derivative)
                self.list_of_layers.append(al_layer)
        self.list_of_outputs = []
        self.target = target
        self.list_of_layer_errors = []
        self.total_error = None
        self.learning_rate = learning_rate

        # if isinstance(self.list_of_layers[-1], FullyConnectedLayer):
        #     if len(self.target) != self.list_of_layers[-1].output:
        #         raise ValueError("The size of target does not match the size of output of the final layer in the nn")
        # elif isinstance(self.list_of_layers[-1], ActivationLayer):
        #     if len(self.target) != self.list_of_layers[-2].output:
        #         raise ValueError("The size of target does not match the size of output of the final layer in the nn")

    def first_layer(self):
        self.list_of_outputs.append(self.list_of_layers[0].forward_propagation(self.input_data))
        print(f"First layer output: {self.list_of_outputs[-1]}")

    def forward_propagate(self):
        for i, layers in enumerate(self.list_of_layers[1:]):
            if isinstance(layers, FullyConnectedLayer):
                self.list_of_outputs.append(layers.forward_propagation(self.list_of_outputs[i - 1]))
                print(f"Fully connected layer {i + 2} output: {self.list_of_outputs[-1]}")
                print(f"Fully connected layer {i + 2} output shape: {self.list_of_outputs[-1].shape}")
            elif isinstance(layers, ActivationLayer):
                self.list_of_outputs.append(layers.forward_propagation(self.list_of_outputs[i - 1]))
                print(f"Activation layer {i + 2} output: {self.list_of_outputs[-1]}")
                print(f"Activation layer {i + 2} output shape: {self.list_of_outputs[-1].shape}")

    def calculate_error(self, loss_function):
        error = loss_function(self.target, self.list_of_outputs[-1])
        self.total_error = error
        print(f"Total error after forward propagation: {self.total_error}")

    def backward_propagate(self, loss_derivative):
        # Compute the error at the output
        output_error = loss_derivative(self.target, self.list_of_outputs[-1])
        self.list_of_layer_errors.append(output_error)
        print(f"Initial output error shape for backpropagation: {output_error.shape}")

        # Going backwards through each layer
        for i, layers in enumerate(self.list_of_layers[::-1]):
            if isinstance(layers, FullyConnectedLayer):
                error = layers.backward_propagation(self.list_of_layer_errors[-1], self.learning_rate, input=self.list_of_outputs[-1])
                self.list_of_layer_errors.append(error)
                print(f"Error after fully connected layer {len(self.list_of_layers) - i} backpropagation: {self.list_of_layer_errors[-1]}")
                print(f"Shape after fully connected layer {len(self.list_of_layers) - i} backpropagation: {self.list_of_layer_errors[-1].shape}")
            elif isinstance(layers, ActivationLayer):
                error = layers.backward_propagation(self.list_of_layer_errors[-1], self.learning_rate)
                self.list_of_layer_errors.append(error)
                print(f"Error after activation layer {len(self.list_of_layers) - i} backpropagation: {self.list_of_layer_errors[-1]}")
                print(f"Shape after activation layer {len(self.list_of_layers) - i} backpropagation: {self.list_of_layer_errors[-1].shape}")


# A model architecture class to define the order of layers in the model. The model architecture will be a list with
# dictionaries in the shape: [{'type': 'FC', 'layer_size': 10, 'output_size': 5}, {'type': 'AL', 'size': 10,
# 'output_size': 10, 'activation_function': 'sigmoid'}] Each item will be a layer. Each item has details about the
# layer.
class ModelArchitecture():
    def __init__(self, input):
        self.architecture = input
        self.layer_type = [i['type'] for i in self.architecture]
        self.layer_size = [i.get('layer_size', None) for i in self.architecture]
        self.layer_output_size = [i.get('output_size', None) for i in self.architecture]

architecture = ModelArchitecture([
    {'type': 'FC', 'layer_size': 2, 'output_size': 3},
    {'type': 'AL', 'output_size': 3},
    {'type': 'FC', 'layer_size': 3, 'output_size': 1},
    {'type': 'AL', 'output_size': 1}
])

input_data = np.array([1, 0.45])
target_data = np.array([1])

nn = NNCompiler(input_size=5, n_layers=4, activation_function=sigmoid,
                activation_derivative=sigmoid_derivative, input_data=input_data,
                model_architecture=architecture, epochs=2, target=target_data, learning_rate=0.01)

# nn = NNCompiler(input_size=2, n_layers=4, activation_function=sigmoid,
#                 activation_derivative=sigmoid_derivative, input_data=np.array([0.5, 0.6]),
#                 model_architecture=architecture, epochs=10, target=np.array([0.7]), learning_rate=0.01)

nn.first_layer()
nn.forward_propagate()
nn.calculate_error(loss_function=mean_squared_error)
nn.backward_propagate(loss_derivative=mse_derivative)
