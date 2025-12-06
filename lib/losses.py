import numpy as np


class MSELoss:
    """
    @brief Mean Squared Error (MSE) loss function.

    Attributes
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Methods
    -------
    __call__(y_true, y_pred)
        Make the MSELoss instance callable.
    backward()
        Compute the gradient of the loss with respect to predictions.
    """
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def __call__(self, y_true, y_pred):
        """
        @brief Make the MSELoss instance callable. Computes the MeanSquaredError (MSE) loss.

        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            MeanSquaredError (MSE) loss.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        return (1 / 2) * np.mean((y_true - y_pred) ** 2)

    def backward(self):
        """
        @brief Compute the gradient of the loss with respect to predictions.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to predictions.
        """
        N = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / N