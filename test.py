import numpy as np

a = np.random.randn(3, 2)
layers = [1, 2, 3, 4, 6]
print({"layer" + str(l) : layers[l] for l in range(len(layers))})