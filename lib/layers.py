import numpy as np


class Layer:
    """
    @brief Base class for layers in a neural network.

    Attributes
    ----------
    trainable : bool
        Indicates if the layer has trainable parameters.
    params : list
        List of layer parameters.
    grads : list
        List of gradients corresponding to the parameters.

    Methods
    -------
    forward(X)
        Perform the forward pass.
    backward(dout)
        Perform the backward pass.
    get_params()
        Retrieve the layer parameters.
    get_grads()
        Retrieve the gradients of the layer parameters.
    """
    def __init__(self):
        """
        @brief Initialize the Layer base class.
        @brief By default, layers are non-trainable with empty params and grads lists.
        """
        self.trainable = False
        self.params = []
        self.grads = []
    
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, dout): 
        raise NotImplementedError
    
    def get_params(self):
        """
        Returns
        -------
        list [W, b]
            List of layer parameters.
        """
        return self.params
    
    def get_grads(self):
        """
        Returns
        -------
        list [dW, db]
            List of gradients corresponding to the layer parameters.
        """
        return self.grads


class Dense(Layer):
    """
    @brief Fully connected layer (Dense layer) implementing a linear transformation.

    Attributes
    ----------
    W : np.ndarray
        Weight matrix of shape (in_features, out_features).
    b : np.ndarray
        Bias vector of shape (1, out_features).
    X : np.ndarray
        Input cache for backward pass.
    Z : np.ndarray
        Output cache for backward pass.
    dW : np.ndarray
        Gradient of weights.
    db : np.ndarray
        Gradient of biases.

    Methods
    -------
    forward(X)
        Perform the forward pass.
    backward(dout)
        Perform the backward pass.
    """
    def __init__(self, in_features, out_features):
        """
        @brief Initialize the Dense layer with random weights and zero biases.
        @brief All weights are initialized from a normal distribution with mean 0 and std 1.0.
        
        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features (Neurons).

        Returns
        -------
        None
        """
        super().__init__()

        self.trainable = True

        self.W = np.random.randn(in_features, out_features) * 1.0     # Standard normal initialization
        self.b = np.zeros((1, out_features))

        self.params = [self.W, self.b]

        # Cached vars for backward
        self.X = None
        self.Z = None
        
    def forward(self, X):
        """
        @brief Forward pass through the Dense layer computing the linear transformation.
        @brief Z = XW + b

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (N, in_features).

        Returns
        -------
        np.ndarray
            Output data of shape (N, out_features).
        """
        self.X = X
        self.Z = X @ self.W + self.b
        return self.Z

    def backward(self, dout):
        """
        @brief Backward pass through the Dense layer computing gradients.
        @brief Computes gradients with respect to inputs, weights, and biases.

        Parameters
        ----------
        dout : np.ndarray
            Upstream gradient of shape (N, out_features).

        Returns
        -------
        np.ndarray
            Gradient with respect to inputs of shape (N, in_features).
        """
        dW = self.X.T @ dout
        db = np.sum(dout, axis=0, keepdims=True)
        dX = dout @ self.W.T

        self.grads = [dW, db]
        
        return dX