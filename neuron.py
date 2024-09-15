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
    future_units: List[LayerUnit]
    future_weights: List[float]

    def __init__(self, identifier=0, value=None):
        super().__init__(identifier, value)

        self.future_units = list()
        self.future_weights = list()

    def __repr__(self):
        return f'BiasUnit({self.id}: {self.value})'


class Neuron(LayerUnit):
    future_units: List[LayerUnit]
    future_weights: List[float]

    past_units: List[LayerUnit]
    past_weights: List[float]

    def __init__(self, identifier=0, value=None):
        super().__init__(identifier, value)

        self.future_units = list()
        self.future_weights = list()

        self.past_units = list()
        self.past_weights = list()

    def __repr__(self):
        return f'Neuron({self.id}: {self.value})'


@dataclass
class Layer:
    units: list = default  # Do not add only neurons, as we have to add a bias unit for each layer.

    def __init__(self, neuron_quantity: int, bias_unit: BiasUnit=None):
        """The size doesn't count the bias unit. The bias unit is always added as the 0 index unit."""

        bias_unit = BiasUnit(0) if bias_unit else None

        self.units = [bias_unit]

        for i in range(1, neuron_quantity + 1):
            self.units.append(Neuron(i))


class NeuralNetwork:
    layers: list


class FullyConnectedNeuralNetwork(NeuralNetwork):

    def __init__(self, *layers):
        self.layers = list()

        for layer in layers:
            self.layers.append(layer)

    def __repr__(self):
        representation = '(' + self.__class__.__name__ + ':\n'

        for layer in self.layers:
            representation += f'\t{layer}\n'

        return representation + ')'


    @staticmethod
    def connect(src: Layer, dst: Layer):
        """Connects every unit in src to every unit in dst.
        The bias unit connects to every other unit in dst but the bias unit in it."""

        # Connect the bias unit.
        for unit in dst.units[1:]:
            src.units[0].future_units.append(unit)
            unit.past_units.append(src.units[0])

        # Connect neuron units.
        for src_unit in src.units[1:]:
            for dst_unit in dst.units[1:]:
                src_unit.future_units.append(dst_unit)
                dst_unit.past_units.append(src_unit)


if __name__ == '__main__':
    initial_layer = Layer(2, BiasUnit())
    hidden_layer = Layer(1, BiasUnit())
    output_layer = Layer(1)

    FullyConnectedNeuralNetwork.connect(initial_layer, hidden_layer)
    FullyConnectedNeuralNetwork.connect(hidden_layer, output_layer)

    print(initial_layer.units)
    print()

    print("Initial layer:")

    for unit in initial_layer.units:
        print(unit)
        print(unit.future_units)

    print()
    print("Hidden layer:")

    for unit in hidden_layer.units[1:]:
        print(unit)
        print(unit.past_units)
        print()

    print(hidden_layer.units)
    print()

    print(output_layer.units)
    print()

    print(initial_layer.units[0].future_units)
    print()

    print(hidden_layer.units[1].past_units)
    print(output_layer.units[1].past_units)

    print(FullyConnectedNeuralNetwork(initial_layer, hidden_layer, output_layer))