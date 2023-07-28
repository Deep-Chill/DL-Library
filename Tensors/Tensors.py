import numpy as np

class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None

    def __str__(self):
        return str(self.data)

    #Returning everything as a Tensor class object.

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        #Element-wise multiplication
        return Tensor(self.data * other.data)

    def matmul(self, other):
        #Matrix Multiplication
        return Tensor(np.dot(self.data, other.data))

    def sum(self):
        return np.sum(self.data)

    def mean(self):
        return np.mean(self.data)

    def max(self):
        return np.max(self.data)

    def argmax(self):
        return np.argmax(self.data)

    def shape(self):
        return self.data.shape




t1 = Tensor([1, 2, 3])
t2 = Tensor([4, 5, 6])

t3 = t1 - t2
print(t3)
print(type(t3))