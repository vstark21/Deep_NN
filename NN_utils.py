import numpy as np

# To make sure that given data is same randomly generated values must be same
np.random.seed(0)

class Layer:

    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = np.zeros((units, 1))
        self.pres_activations = None
        self.prev_activations = None

    def forward_prop(self, A_prev):
        Z_pres = np.dot(self.weights, A_prev) + self.bias
        if self.activation == "relu":
            A_pres = np.maximum(Z_pres, 0)
        elif self.activation == "sigmoid":
            A_pres = 1/(1 + np.exp(-Z_pres))
        elif self.activation == "softmax":
            pass
        else:
            A_pres = Z_pres
        self.pres_activations = A_pres
        self.pres_activations = A_prev
        return A_pres

    def backward_prop(self, dA_pres):
        m = dA_pres.shape[-1]
        if self.activation == None:
            db = np.sum(dA_pres, axis=1, keepdims=True) * 1 / m
            dW = np.dot(dA_pres, self.pres_activations.T) * 1 / m
            dA_prev = np.dot(self.weights.T, dA_pres)
        return dW, dA_prev, db
    
    def update_parameters(self, dW, db, learning_rate):
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
    
class NN_model:

    def __init__(self, input_size, layers=[], loss_type="mse", num_layers=None):
        self.layers = {"layer" + str(l + 1) : layers[l] for l in range(len(layers))}
        self.loss = Loss(loss_type)
        self.input_size = input_size
        self.num_layers = len(layers)
        self.layers["layer1"].weights = np.random.randn(self.layers["layer1"].units, input_size)
        for l in range(1, self.num_layers):
            self.layers["layer" + str(l + 1)].weights = np.random.randn(self.layers["layer" + str(l + 1)].units, self.layers["layer" + str(l)].units) * 0.01
    
    def fit(self, X_train, Y_train, epochs=10, learning_rate=0.001):
        
        for epoch in range(epochs):
            A_pres = X_train
            for l in range(self.num_layers):
                
                A_pres = self.layers["layer" + str(l + 1)].forward_prop(A_pres)

            dA = self.loss.back_prop(A_pres, Y_train)
            
            for l in range(self.num_layers, 0, -1):
                
                dW, dA, db = self.layers["layer" + str(l)].backward_prop(dA)
                
                try:
                    self.layers["layer" + str(l)].update_parameters(dW, db, learning_rate)
                except Exception as e:
                    print(epoch, e)
                    quit()
            



    # Needed to be changed
    def summary(self):
        for layer in self.layers.values():
            print(layer.units, layer.weights, layer.bias)
    
    # Needed to be changed
    def predict(self, A_prev):
        for layer in self.layers.values():
            A_prev = layer.forward_prop(A_prev)
        return A_prev
        

class Loss:

    def __init__(self, loss_type="mse"):
        self.loss_type = loss_type
    
    def back_prop(self, y_pred, y_true):
        if self.loss_type == "mse":
            dA = 2 * (y_pred - y_true)
        return dA



