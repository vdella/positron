from neuron import NeuralNetwork
from exception import IrregularSizeForMatrixComparisonException
from math import log


def hypothesis_for(weights, units):
    if len(weights) != len(units):
        raise IrregularSizeForMatrixComparisonException('Weights and units, a.k.a. x, must have the same size.')

    return sum([weights[i] * units[i] for i in range(len(weights))])


def loss_for(network, weights, x, y, lambda_factor=1):  # TODO: test.
    if len(weights) != len(x):
        raise IrregularSizeForMatrixComparisonException('Weights and x must have the same size.')

    m = len(x)

    reduce_units = 0
    for i in range(m):
        for k in range(len(y)):
            reduce_units += y[k] * log(hypothesis_for(weights, x)) + (1 - y[k]) * log(1 - hypothesis_for(weights, x))

    reduce_weights = 0
    for l in range(len(network.layers - 1)):
        for i in range(len(network.layers[l])):
            for j in range(len(network.layers[l + 1])):
                reduce_weights += network.layers[l][i].future_weights[j] ** 2

    return (-1 / m) * reduce_units + (lambda_factor / (2 * m)) * reduce_weights


def gradient_for(weights, x, y):  # TODO
    return 0.0


def forward_propagate(network: NeuralNetwork):  # TODO
    return 1


def back_propagate(network: NeuralNetwork):  # TODO
    return 1
