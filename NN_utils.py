import numpy as np

# To make sure that given data is same randomly generated values must be same
np.random.seed(0)



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
            pass

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
        return dW, dA_prev, db
    

    def update_parameters(self, dW, db, learning_rate):

        """
            Updating parameters of current layer

        """
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

    

# NN_model class for Neural Network
class NN_model:

    def __init__(self, input_size, layers=[], loss_type="mse", num_layers=None):

        """
            Initializing the attributes of Neural Layer:

            input_size : size of a training example
            layers : Layers which are to be used in sequence (Class of layer must be Layer)
            loss_type : type of loss you want to use ("mse", "binary_crossentropy", ---)
            num_layers : Number of layers in Neural Network (Not compulsary)

        """

        self.layers = {"layer" + str(l + 1) : layers[l] for l in range(len(layers))}
        self.loss = Loss(loss_type)
        self.input_size = input_size
        self.num_layers = len(layers)

        self.layers["layer1"].weights = np.random.randn(self.layers["layer1"].units, input_size)
        for l in range(1, self.num_layers):
            self.layers["layer" + str(l + 1)].weights = np.random.randn(self.layers["layer" + str(l + 1)].units, self.layers["layer" + str(l)].units) * 0.01


    def fit(self, X_train, Y_train, epochs=10, learning_rate=0.001):
        
        """
            This function fits the data to the given labels

            X_train : Features which are to be trained
            Y_train : Labels which are to be trained
            epochs : Number of passes through the training set
            learning_rate : rate at which gradient descent works

        """
        for epoch in range(epochs):

            A_pres = X_train
            for l in range(self.num_layers):
                A_pres = self.layers["layer" + str(l + 1)].forward_prop(A_pres)

            dA = self.loss.back_prop(A_pres, Y_train)
            
            for l in range(self.num_layers, 0, -1):
                dW, dA, db = self.layers["layer" + str(l)].backward_prop(dA)
                
                self.layers["layer" + str(l)].update_parameters(dW, db, learning_rate)

        train_loss, train_accuracy = self.compute_loss_acc(A_pres, Y_train)
        print("Total training loss : {0} and accuracy on training set : {1}".format(train_loss, train_accuracy))
                       


    # To get predicted output
    def predict(self, A0):

        for l in range(self.num_layers):
            A0 = self.layers["layer" + str(l + 1)].forward_prop(A0)
        
        return A0


    # Computing training loss and accuracy at the end of training
    def compute_loss_acc(self, Y_pred, Y_true):
        
        if self.loss.loss_type == "binary_crossentropy":
            loss = np.squeeze(-1 * np.sum(np.add(np.multiply(1 - Y_true, np.log(1 - Y_pred)), np.multiply(Y_true, np.log(Y_pred))), axis=1))
            Y_pred_round = np.round(Y_pred, 0)
            accuracy, m = 0, len(Y_true)
            for i in range(m):
                if Y_true[i] == Y_pred_round[0][i]:
                    accuracy += 1
        return loss / m, accuracy / m


    # TO get summary (details) of the model
    def summary(self):
        
        for l in range(self.num_layers):
            print(self.layers["layer" + str(l + 1)].units, self.layers["layer" + str(l + 1)].weights, self.layers["layer" + str(l + 1)].bias)


    # To Evaluate model on validation data
    def evaluate(self, X, Y):

        Y_pred = self.predict(X)
        loss, accuracy = self.compute_loss_acc(Y_pred, Y)
        print("Loss : {0} and Accuracy : {1}".format(np.round(loss, 5), accuracy))
    
        

class Loss:

    def __init__(self, loss_type="mse"):
        self.loss_type = loss_type
    
    def back_prop(self, y_pred, y_true):
        if self.loss_type == "mse":
            dA = 2 * (y_pred - y_true)
        elif self.loss_type == "binary_crossentropy":
            dA = np.divide(y_pred - y_true, y_pred - np.power(y_pred, 2))
        return dA
