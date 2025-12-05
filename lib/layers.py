import numpy as np


class Layer:
    """
    @brief Base class for layers in a neural network.

    Attributes
    ----------
    trainable : bool
        Indicates if the layer has trainable parameters.

    Methods
    -------
    forward(X)
        Perform the forward pass.
    backward(dout)
        Perform the backward pass.
    """
    def __init__(self):
        """
        @brief Initialize the Layer base class.
        @brief By default, layers are non-trainable with empty params and grads lists.
        """
        self.trainable = False
    
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, dout): 
        raise NotImplementedError


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
    def __init__(self, in_features, out_features, scale=0.01):
        """
        @brief Initialize the Dense layer with random weights and zero biases.
        @brief All weights are initialized from a normal distribution with mean 0 and std 1.0.
        
        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features (Neurons).
        scale : float, optional
            Scaling factor for weight initialization, by default 0.01.

        Returns
        -------
        None
        """
        super().__init__()

        self.trainable = True

        self.W = np.random.randn(in_features, out_features) * scale     # Standard normal initialization scaled to small values
        self.b = np.zeros((1, out_features))
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

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
        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        dX = dout @ self.W.T
        
        return dX