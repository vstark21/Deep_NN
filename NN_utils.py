import numpy as np


class Layer:

    def __init__(self, units, activation="relu", input_shape):
        self.units = units
        self.activation = activation

