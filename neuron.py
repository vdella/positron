from email.policy import default
from typing import List
from dataclasses import dataclass
import numpy as np


class LayerUnit:
    value: float
    id: int

    def __init__(self, identifier=0, value=None):
        self.id = identifier
        self.value = value if value else np.random.rand() * 0.1


class BiasUnit(LayerUnit):
    future: List[LayerUnit]

    def __init__(self):
        super().__init__()
        self.future = list()

    def __repr__(self):
        return f'BiasUnit({self.id}: {self.value})'


class Neuron(LayerUnit):
    future: List[LayerUnit]
    past: List[LayerUnit]

    def __init__(self):
        super().__init__()
        self.future = list()
        self.past = list()

    def __repr__(self):
        return f'Neuron({self.id}: {self.value})'


@dataclass
class Layer:
    units: list = default  # Do not add only neurons, as we have to add a bias unit for each layer.

    def __init__(self, size: int):
        bias_unit = BiasUnit(0)

        self.units = [bias_unit]

        for i in range(1, size):
            self.units.append(Neuron(i))


class NeuralNetwork:
    layers: list
    weights: list


class FullyConnectedNeuralNetwork(NeuralNetwork):

    def __init__(self, input_layer_size: int, hidden_layer_sizes: List[int], output_layer_size: int):
        self.layers = list()
        self.layers.append(Layer(input_layer_size))

    @staticmethod
    def sparse_connect(src: Layer, dst: Layer):
        src.units[0]

        for src_unit in src.units:
            for dst_unit in dst.units:
                src_unit.futures.append(dst_unit)
                dst_unit.pasts.append(src_unit)


if __name__ == '__main__':
    print(Layer(5).units[0])