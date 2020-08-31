from NN_utils import *
from data_gen import *
import numpy as np

model = NN_model(input_size=3, layers=[
    Layer(2, activation="relu"),
    Layer(1, activation="sigmoid")
], loss_type="binary_crossentropy")

m = 10000
X, Y = data(m)
model.fit(X, Y, epochs=20000)
x, y = data(m)

model.evaluate(x, y)