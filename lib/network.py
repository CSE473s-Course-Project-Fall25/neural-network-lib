### Network module for custom NN library implementation ###
from .losses import MSELoss
from .optimizer import SGDOptimizer


class Sequential:
    """
    @brief A simple feedforward neural network class.

    Attributes
    ----------
    layers : list
        List of layers in the network.
    loss : MSELoss
        Loss function for the network.

    Methods
    -------
    add(layer)
        Add a layer to the network.
    compile(loss)
        Compile the network with a specified loss function.
    forward(X)
        Perform a forward pass through the network.
    backward(y_true)
        Perform a backward pass through the network.
    fit(X, y, epochs, learning_rate)
        Train the network on the provided data.
    predict(X)
        Make predictions using the trained network.
    """
    def __init__(self, layers=None):
        """
        @brief Initialize the Sequential neural network.
        
        Parameters
        ----------
        layers : list, optional
            List of layers to initialize the network with (default is None).
        """
        self.layers = layers if layers is not None else []
        self.loss = None

    def add(self, layer):
        """
        @brief Add a layer to the network.
        
        Parameters
        ----------
        layer : Layer
            Layer to be added to the network.
        """
        self.layers.append(layer)

    def forward(self, X):
        """
        @brief Perform a forward pass through the network.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
            
        Returns
        -------
        np.ndarray
            Output of the network after the forward pass.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_grad):
        """
        @brief Perform a backward pass through the network.

        Parameters
        ----------
        y_grad : np.ndarray
            Gradient of the loss with respect to the network output.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the network input.
        """
        dout = y_grad
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def train_step(self, X, y, loss_fn: MSELoss, opt: SGDOptimizer):
        """
        @brief Perform a single training step: forward pass, loss computation, backward pass, and parameter update.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            True target values.
        opt : SGDOptimizer
            Optimizer to update the network parameters.
        """
        # Forward pass
        y_pred = self.forward(X)

        # Compute loss & initial gradient
        loss = loss_fn(y, y_pred)
        dout = loss_fn.backward()

        # Backpropagation
        self.backward(dout)

        opt.step(self.layers)

        return loss
    
    def fit(self, X, y, loss_fn = MSELoss(), opt = SGDOptimizer(learning_rate=1.0), epochs=1000):
        """
        @brief Train the network on the provided data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            True target values.
        epochs : int, optional
            Number of training epochs (default is 1000).
        loss_fn : Loss (Optional)
            Loss function to use for training.
        opt : Optimizer (Optional)
            Optimizer to update the network parameters.
        """
        history = []
        for epoch in range(epochs):
            loss = self.train_step(X, y, loss_fn, opt)
            history.append(loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        return history