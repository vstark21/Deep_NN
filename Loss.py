import numpy as np

# Loss helper class used in neural network
class Loss:

    def __init__(self, loss_type="mse"):

        self.loss_type = loss_type
    


    def back_prop(self, y_pred, y_true):

        """
            This function starts back prop

        """
        
        if self.loss_type == "mse":
            dA = 2 * (y_pred - y_true)
        
        elif self.loss_type == "binary_crossentropy":
            dA = np.divide(y_pred - y_true, y_pred - np.power(y_pred, 2))
        
        elif self.loss_type == "categorical_crossentropy":
            dA = y_pred - y_true
        
        return dA