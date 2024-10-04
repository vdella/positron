import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    """the softmax applies the standard exponential function to each element z[i] of the input vector z
    (consisting of K real numbers), and normalizes these values by dividing by the sum of all these exponentials."""
    exps = np.exp(x - x.max(axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


hidden_activation = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative)
}
output_activation = {'sigmoid': sigmoid, 'softmax': softmax}
