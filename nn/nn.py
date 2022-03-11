# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.
    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 
            'activation:': 'sigmoid'}] will generate a 2 layer deep fully connected network with an 
            input dimension of 64, a 32 dimension hidden layer and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.
    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch: List[Dict[str, int]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!
        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.
        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.
        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.
        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # want to take activation function of Z, which is w_curr * A_prev + b_curr
        Z_curr = W_curr @ A_prev + b_curr
        
        # choose which activation to use to get A
        if activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        
        if activation == 'relu':
            A_curr = self._relu(Z_curr)
    
        return Z_curr, A_curr

 
    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.
        We want to obtain A and Z for each layer, and save the results.
        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].
        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        
     
        cache = {}
        A_curr = X # first layer just assign X to be activation
        layer_idx=1 # starting at first layer
        # itterate through the layers, using single_forward method:
        activation_list = [layer['activation'] for layer in self.arch]
        for activation in activation_list:
            A_prev = A_curr
            W_curr = self._param_dict['W' + str(layer_idx)] # get W
            b_curr = self._param_dict['b' + str(layer_idx)] # get b
            # use single forward to get A and Z
            Z_curr, A_curr = self._single_forward(W_curr, b_curr, A_prev, activation)
            cache['Z' + str(layer_idx)] = Z_curr # store Z_curr
            cache['A' + str(layer_idx-1)] = A_prev # store A_prev
            layer_idx+=1 # move to next layer for next loop
            
        return A_curr, cache # A_curr is final y_hat
        
    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.
        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.
        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        # first we must obtain dZ_curr, using the derivative of the activation function
        # basically, we will be getting dC/dZ from dA/dZ * dC/dA, which is the local gradiant
        if  activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        if  activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        
        m = A_prev.shape[1] # number of observations
        
        # get dA_prev/dC
        dA_prev = np.dot(W_curr.T, dZ_curr)
    
        # Now we want dC/dW = 1/m*dC/dZ * A_prev: 
        dW_curr = (1/m)*np.dot(dZ_curr, A_prev.T)
        
        # then, dC/db = 1/m*sum(dC/dZ)
        db_curr = (1/m)*sum(dZ_curr[0])
        db_curr = np.array([db_curr]).reshape(1,-1) # has to be this shape
        
        return dA_prev, dW_curr, db_curr
       

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.
        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.
        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        
        # initialize dictionary of gradient info
        grad_dict = {}
        
        # getting dA/dC to start
        dA_prev = self._loss_function_backprop(y, y_hat)
        
        # for each layer, we much calculate W, b, X, A prev, dA, and send
        # this to single backprop, with the correct activation function for that layer
        activation_list = [layer['activation'] for layer in self.arch]
        layer_idx=len(activation_list) # starting at last layer
        for activation in np.array(activation_list)[::-1]:
            # we are going BACKWARDS through the layers 
 
            dA_curr = dA_prev

            # obtain values to use
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(layer_idx-1)] 
            dA_curr = dA_prev

            # Now we have A_prev, Z_curr, W_curr, and b_curr: 
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, 
                                                              b_curr, 
                                                              Z_curr, 
                                                              A_prev, 
                                                              dA_curr, 
                                                              activation)
            # saving gradients
            grad_dict['dW' + str(layer_idx)] = dW_curr # dC/dW
            grad_dict['db' + str(layer_idx)] = db_curr # db/dW
            layer_idx-=1 # then go to previous layer for next loop
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything
        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        Returns:
            None
        """
            
        for idx in range(1,len(self.arch)+1):
            # update W with gradiant
            curr_W_val = self._param_dict['W' + str(idx)]
            grad_W = grad_dict['dW' + str(idx)]
            self._param_dict['W' + str(idx)] = (curr_W_val - self._lr * grad_W)

            # update b with gradiant
            curr_b_val = self._param_dict['b' + str(idx)]
            grad_b = grad_dict['db' + str(idx)]
            self._param_dict['b' + str(idx)] = (curr_b_val - self._lr * grad_b)
            
        return None


    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.
        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        
        for idx in np.arange(self._epochs): # itterate through each epoch
            # use training data to get y_hat and find loss for training
            y_hat_train, cache_train = self.forward(X_train) 
            train_loss = self._loss_function(y_train, y_hat_train) # get loss for this
            per_epoch_loss_train.append(train_loss) # append to loss list
            
            # use validation data to get y_hat and find loss for validation
            y_hat_val, cache_val = self.forward(X_val)
            val_loss = self._loss_function(y_val, y_hat_val)
            per_epoch_loss_val.append(val_loss)
            # finally, update params
            
            grad_dict = self.backprop(y_train, y_hat_train, cache_train) # get gradiant dict
            self._update_params(grad_dict) # add to param dict     
        
        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.
        Args:
            X: ArrayLike
                Input data for prediction.
        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, cache = self.forward(X.T)
        return y_hat.T

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.
        Args:
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        
        nl_transform = 1/(1+np.exp(-Z))    
        return nl_transform

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.
        Args:
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = Z*(Z > 0)
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.
        
        dA/dZ = sigmoid'(Z)
        dC/dZ = dC/dA * dA/dZ
        dC/dZ = sigmoid'(Z) * dA
        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix. (dC/dZ)
        """
        derivative = self._sigmoid(Z)*(1-self._sigmoid(Z))
        dZ = derivative*dA
        return dZ
        
    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.
        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.
        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        derivative = (1*(Z > 0))
        dZ = derivative*dA
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.
        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.
        Returns:
            loss: float
                Average loss over mini-batch.
        """
        
        N = y_hat.shape[1]
        
        # assures we don't get any errors for dividing by 0!
        y_hat=np.array([yi if yi!=0 else .0000001 for yi in y_hat[0]])
        y_hat=np.array([yi if yi!=1 else .9999999 for yi in y_hat])
        y_hat=np.array(y_hat).reshape(1,-1)
        # calculate the binary cross entropy loss
 
        loss = (-1/N)*sum(
                            [(yi*np.log(ypi)+(1-yi)*np.log(1-ypi)) 
                            for yi, ypi in zip(y, y_hat)][0] 
                          )

        return loss


    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.
        dC/dA = 1/m*[-y/y_hat+(1-y)/(1-y_hat)]
        
        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.
        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = (-y/y_hat)+(1-y)/(1-y_hat)
        
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.
        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """

        loss = np.mean((y - y_hat)**2)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.
        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.
        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = -2*(y-y_hat)
        return dA
        
    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.
        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """

        if self._loss_func == 'bce':
            loss = self._binary_cross_entropy(y, y_hat)
            
        if self._loss_func == 'mse':
            loss = self._mean_squared_error(y, y_hat)

        return loss  

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        if self._loss_func == 'bce':
            dA = self._binary_cross_entropy_backprop(y, y_hat)
        
        if self._loss_func == 'mse':
            dA = self._mean_squared_error_backprop(y, y_hat)
        
        return dA