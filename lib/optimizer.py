class SGDOptimizer:
    """
    @brief Stochastic Gradient Descent (SGD) optimizer.

    Attributes
    ----------
    learning_rate : float
        Learning rate for the optimizer.

    Methods
    -------
    step(params, grads)
        Update parameters using gradients.
    """
    def __init__(self, learning_rate=1.0):
        """
        @brief Initialize the SGD optimizer with a specified learning rate.

        Parameters
        ----------
        learning_rate : float, optional
            Learning rate for the optimizer (default is 1.0).
        """
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        @brief Update parameters using gradients.

        Parameters
        ----------
        layers : list
            List of layers in the neural network.
        """
        for layer in layers:
            if layer.trainable:
                for i in range(len(layer.params)):
                    layer.params[i] -= self.learning_rate * layer.grads[i]