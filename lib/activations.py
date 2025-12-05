import numpy as np
from .layers import Layer


class ReLU(Layer):
    """
    @brief Rectified Linear Unit (ReLU) activation function layer.
    
    Attributes
    ----------
    mask : np.ndarray
        Mask of input values greater than zero for backward pass.

    Methods
    -------
    forward(X)
        Perform the forward pass applying ReLU activation.
    backward(dout)
        Perform the backward pass computing gradients.
    """
    def forward(self, X):
        """
        @brief Perform the forward pass applying ReLU activation.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
            
        Returns
        -------
        np.ndarray
            Output data after applying ReLU.
        """
        self.mask = X > 0
        return X * self.mask
    
    def backward(self, dout):
        """
        @brief Perform the backward pass computing gradients.
        
        Parameters
        ----------
        dout : np.ndarray
            Upstream gradient.
        
        Returns
        -------
        np.ndarray
            Gradient with respect to input.
        """
        return dout * self.mask


class Sigmoid(Layer):
    """
    @brief Sigmoid activation function layer.
    
    Attributes
    ----------
    Y : np.ndarray
        Output after applying sigmoid activation.

    Methods
    -------
    forward(X)
        Perform the forward pass applying sigmoid activation.
    backward(dout)
        Perform the backward pass computing gradients.
    """
    def forward(self, X):
        """
        @brief Perform the forward pass applying sigmoid activation.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
            
        Returns
        -------
        np.ndarray
            Output data after applying sigmoid.
        """
        self.Y = 1 / (1 + np.exp(-X))
        return self.Y
    
    def backward(self, dout):
        """
        @brief Perform the backward pass computing gradients.
        
        Parameters
        ----------
        dout : np.ndarray
            Upstream gradient.
        
        Returns
        -------
        np.ndarray
            Gradient with respect to input.
        """
        return dout * self.Y * (1 - self.Y)


class Tanh(Layer):
    """
    @brief Hyperbolic Tangent (Tanh) activation function layer.
    
    Attributes
    ----------
    Y : np.ndarray
        Output after applying tanh activation.

    Methods
    -------
    forward(X)
        Perform the forward pass applying tanh activation.
    backward(dout)
        Perform the backward pass computing gradients.
    """
    def forward(self, X):
        """
        @brief Perform the forward pass applying tanh activation.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
            
        Returns
        -------
        np.ndarray
            Output data after applying tanh.
        """
        self.Y = np.tanh(X)
        return self.Y
    
    def backward(self, dout):
        """
        @brief Perform the backward pass computing gradients.
        
        Parameters
        ----------
        dout : np.ndarray
            Upstream gradient.
        
        Returns
        -------
        np.ndarray
            Gradient with respect to input.
        """
        return dout * (1 - self.Y ** 2)
    

class Softmax(Layer):
    """
    @brief Softmax activation function layer.
    
    Attributes
    ----------
    Y : np.ndarray
        Output after applying softmax activation.

    Methods
    -------
    forward(X)
        Perform the forward pass applying softmax activation.
    backward(dout)
        Perform the backward pass computing gradients.
    """
    def forward(self, X):
        """
        @brief Perform the forward pass applying softmax activation.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
            
        Returns
        -------
        np.ndarray
            Output data after applying softmax.
        """
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.Y = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.Y
    
    def backward(self, dout):
        """
        @brief Perform the backward pass computing gradients.
        
        Parameters
        ----------
        dout : np.ndarray
            Upstream gradient.
        
        Returns
        -------
        np.ndarray
            Gradient with respect to input.
        """
        # Note: The full Jacobian is not computed here for efficiency.
        # shortcut when softmax is paired with cross-entropy loss.
        return dout 