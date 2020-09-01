import numpy as np

# Layer class for hidden and output layers in Neural Network
class Layer:

    def __init__(self, units, activation=None):

        """
            Initializing the attributes of Neural Layer:

            units : Number of Units in Layer
            activation : Activation of Layer ("relu", "sigmoid", "softmax", ---)
            weights : As shape of weights depend upon previous layer shape we assign them in model initialization
            bias : whereas bias depends on units in current layer
            pres_activations : Activations of this layer
            prev_activations : Activations of previous layer

        """

        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = np.zeros((units, 1))
        self.pres_activations = None
        self.prev_activations = None


    def forward_prop(self, A_prev):

        """
            Implementing forward propagation through this layer

            A_prev : Activations of Previous layer
            Z_pres : Linear Activation them Non-linear activation is applied based activation type of the current layer

        """

        Z_pres = np.dot(self.weights, A_prev) + self.bias

        if self.activation == "relu":
            A_pres = np.maximum(Z_pres, 0)

        elif self.activation == "sigmoid":
            A_pres = 1/(1 + np.exp(-Z_pres))

        elif self.activation == "softmax":
            exp_Z = np.exp(Z_pres - np.max(Z_pres, axis=0))
            sum_exp_Z = np.sum(exp_Z, axis=0)
            A_pres = np.divide(exp_Z, sum_exp_Z)

        else:
            A_pres = Z_pres
        
        self.pres_activations = A_pres
        self.prev_activations = A_prev

        return A_pres


    def backward_prop(self, dA_pres):

        """
            Implementing backward propogation through this layer

            dA_pres : change in cost function with respect to activations of current layer

        """
        m = dA_pres.shape[-1]

        if self.activation == None:
            db = np.sum(dA_pres, axis=1, keepdims=True) * 1 / m
            dW = np.dot(dA_pres, self.prev_activations.T) * 1 / m
            dA_prev = np.dot(self.weights.T, dA_pres)
        
        elif self.activation == "relu":
            dZ = dA_pres.copy()
            Z_pres = np.dot(self.weights, self.prev_activations) + self.bias
            dZ[Z_pres <= 0] = 0
            dW = np.dot(dZ, self.prev_activations.T) * 1 / m
            db = np.sum(dZ, axis=1, keepdims=True) * 1 / m
            dA_prev = np.dot(self.weights.T, dZ)
        
        elif self.activation == "sigmoid":
            dZ = np.multiply(np.multiply(self.pres_activations, 1 - self.pres_activations), dA_pres)
            dW = np.dot(dZ, self.prev_activations.T) * 1 / m
            db = np.sum(dZ, axis=1, keepdims=True) * 1 / m
            dA_prev = np.dot(self.weights.T, dZ)

        elif self.activation == "softmax":
            dZ = dA_pres
            dW = np.dot(dZ, self.prev_activations.T) * 1 / m
            db = np.sum(dZ, axis=1, keepdims=True) * 1 / m
            dA_prev = np.dot(self.weights.T, dZ)
        return dW, dA_prev, db
    

    def update_parameters(self, dW, db, learning_rate):

        """
            Updating parameters of current layer

        """
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db