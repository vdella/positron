from email.policy import default
from typing import List
from dataclasses import dataclass
import numpy as np


class LayerUnit:
    value: float
    id: int

    def __init__(self, value=None, identifier=0):
        self.value = value if value else np.random.rand() * 0.1
        self.id = identifier


class BiasUnit(LayerUnit):
    future: List[LayerUnit]


class Neuron(LayerUnit):
    future: List[LayerUnit]
    past: List[LayerUnit]


@dataclass
class Layer:
    units: list = default  # Do not add only neurons, as we have to add a bias unit for each layer.

    def __init__(self, size: int):
        bias_unit = BiasUnit(0)

        self.units = [bias_unit]

        for i in range(1, size):
            self.units.append(Neuron(i))


if __name__ == '__main__':
    pass