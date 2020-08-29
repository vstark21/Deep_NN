from NN_utils import *

model = NN_model(input_size=3, layers=[
    Layer(2),
    Layer(1)
])

m = 50
X = np.random.randn(3, m)
Y = np.array([21 * x1 + 10 * x2 - 3 * x3 + np.random.rand() for x1, x2, x3 in zip(X[0], X[1], X[2])])
print(X.shape)
model.summary()
model.fit(X, Y, epochs=5000)
model.summary()
x = np.array([[10], [1], [1]])
print(model.predict(x), 21 * x[0][0] + 10 * x[1][0] - 3 * x[2][0])
# print(Loss().compute_loss(model, X, Y))

from sklearn.linear_model import LinearRegression

lin = LinearRegression().fit(X.T, Y)
print(lin.predict(x.T))