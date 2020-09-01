from NN_utils import *
from data_gen import *
import numpy as np

model = NN_model(input_size=28*28, layers=[
    # Layer(3, activation="relu"),
    Layer(2, activation="softmax")
], loss_type="categorical_crossentropy")

