# Author : V I S H W A S [https://github.com/vstark21]

import numpy as np
from Loss import *
from Layer import *
from ProgressBar import *

# NN_model class for Neural Network
class NN_model:

    def __init__(self, input_size, layers=[], loss_type="mse", num_layers=None, print_loss=False):

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
        self.print_loss = print_loss

        self.layers["layer1"].weights = np.random.randn(self.layers["layer1"].units, input_size) * 0.01
        for l in range(1, self.num_layers):
            self.layers["layer" + str(l + 1)].weights = np.random.randn(self.layers["layer" + str(l + 1)].units, self.layers["layer" + str(l)].units) * 0.01



    def fit(self, X_train, Y_train, epochs=10, learning_rate=0.008):
        
        """
            This function fits the data to the given labels

            X_train : Features which are to be trained
            Y_train : Labels which are to be trained
            epochs : Number of passes through the training set
            learning_rate : rate at which gradient descent works

        """

        progress_bar = ProgressBar(epochs)
        for epoch in range(epochs):

            A_pres = X_train.copy()
            for l in range(self.num_layers):
                A_pres = self.layers["layer" + str(l + 1)].forward_prop(A_pres)

            dA = self.loss.back_prop(A_pres, Y_train)
            
            for l in range(self.num_layers, 0, -1):
                dW, dA, db = self.layers["layer" + str(l)].backward_prop(dA)
                
                self.layers["layer" + str(l)].update_parameters(dW, db, learning_rate)
            
            if self.print_loss:
                progress_bar.update_with_loss(epoch, self.compute_loss_acc(A_pres, Y_train))
            else:
                progress_bar.update(epoch)

        progress_bar.end()
        print("Done Training")

        train_loss, train_accuracy = self.compute_loss_acc(A_pres, Y_train)
        print("Total training loss : {0} and accuracy on training set : {1}".format(train_loss, train_accuracy))
                       


    def predict(self, A0):

        """
            This function predicts final output values for A0

            A0 : Input Units or X

        """

        for l in range(self.num_layers):
            A0 = self.layers["layer" + str(l + 1)].forward_prop(A0)
        
        return A0



    def compute_loss_acc(self, Y_pred, Y_true):

        """
            This function computes loss and accuracy

            Y_pred : Predicted output units by neural network
            Y_true : True output units

        """
        
        if self.loss.loss_type == "categorical_crossentropy":
            
            Y_true_argmax = np.argmax(Y_true, axis=0)
            Y_pred_argmax = np.argmax(Y_pred, axis=0)
            accuracy, loss, m = 0, 0, Y_true.shape[-1]
            for el1, el2, i in zip(Y_true_argmax, Y_pred_argmax, range(m)):
                if el1 == el2:
                    accuracy += 1
                loss -= np.log(Y_pred[el2][i])
            
            return loss / m, accuracy / m

        elif self.loss.loss_type == "binary_crossentropy":
            loss = np.squeeze(-1 * np.sum(np.add(np.multiply(1 - Y_true, np.log(1 - Y_pred)), np.multiply(Y_true, np.log(Y_pred))), axis=1))
            Y_pred_round = np.round(Y_pred, 0)
            accuracy, m = 0, len(Y_true)
            for i in range(m):
                if Y_true[i] == Y_pred_round[0][i]:
                    accuracy += 1

        return loss / m, accuracy / m



    def summary(self):
        
        """
            This function gives the summary of model

        """

        for l in range(self.num_layers):
            print(self.layers["layer" + str(l + 1)].units, self.layers["layer" + str(l + 1)].weights, self.layers["layer" + str(l + 1)].bias)



    def evaluate(self, X, Y):
        
        """
            This function fits the data to the given labels

            X : Features which are to be evaluated
            Y : Labels which are to be evaluated

        """

        Y_pred = self.predict(X)
        loss, accuracy = self.compute_loss_acc(Y_pred, Y)
        print("Loss : {0} and Accuracy : {1}".format(np.round(loss, 5), accuracy))
    
        