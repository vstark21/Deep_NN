from NN_utils import *

model = NN_model(input_size=3, layers=[
    Layer(2, activation="relu"),
    Layer(1)
])

m = 1000
X = np.random.randn(3, m)
Y = np.array([x1 * 20 + 10 * x2 - 3 * x3 + np.random.rand()  for x1, x2, x3 in zip(X[0], X[1], X[2])])
print(X.shape)
model.summary()
model.fit(X, Y, epochs=50000)
model.summary()
x = np.array([[10], [1], [1]])
print(model.predict(x),  x[0][0] * 20 + 10 * x[1][0] - 3 * x[2][0])
