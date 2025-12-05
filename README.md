# neural-network-lib

A simple, custom-built library for creating and training neural networks.

## Repository Structure

The core library components are located within the `lib/` directory.

```
neural-network-lib/
├── lib/
│   ├── __init__.py         # Exports main components
│   ├── activations.py      # Activation functions (ReLU, Sigmoid, Tanh, Softmax)
│   ├── layers.py           # Base Layer and Dense (Fully Connected) layer
│   ├── losses.py           # Loss functions (MSELoss)
│   ├── network.py          # Sequential model class and training logic
│   └── optimizer.py        # Optimizers (SGDOptimizer)
├── notebooks/
│   └── project_demo.ipynb  # Example usage ([XOR] problem)
├── README.md               # This file
└── requirements.txt        # Required packages
```

## Installation

The library is primarily built on **NumPy**. You can install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

The content of `requirements.txt` is:

```
numpy
matplotlib
tensorflow
keras
pandas
scikit-learn
```

## Library Documentation

The library is composed of several modules providing the core building blocks for a neural network.

### Network Module (`network.py`)

| Class | Description |
| :--- | :--- |
| `Sequential` | A simple feedforward neural network class. It manages a list of layers and provides core methods like `forward()`, `backward()`, and `fit()` for training. |

### Layer Module (`layers.py`)

| Class | Description |
| :--- | :--- |
| `Layer` | The base class for all layers. Layers are non-trainable by default and implement `forward()` and `backward()` methods. |
| `Dense` | A **fully connected layer** implementing the linear transformation $Z = XW + b$. It is a trainable layer, storing weights (`W`), biases (`b`), and their respective gradients (`dW`, `db`). |

### Activation Module (`activations.py`)

These layers are typically non-trainable and apply their respective functions element-wise:

  * **`ReLU`**: Rectified Linear Unit activation function.
  * **`Sigmoid`**: Sigmoid activation function $\sigma(x) = \frac{1}{1 + e^{-x}}$.
  * **`Tanh`**: Hyperbolic Tangent activation function.
  * **`Softmax`**: Softmax activation function for output layers.

### Loss Module (`losses.py`)

| Class | Description |
| :--- | :--- |
| `MSELoss` | **Mean Squared Error (MSE)** loss function. It calculates the loss as $L = \text{np.mean}((y_{\text{true}} - y_{\text{pred}}) ^ 2)$ and provides the gradient $\frac{\partial L}{\partial y_{\text{pred}}}$. |

### Optimizer Module (`optimizer.py`)

| Class | Description |
| :--- | :--- |
| `SGDOptimizer` | **Stochastic Gradient Descent (SGD)** optimizer. It updates the trainable parameters of each layer using the rule: $\text{layer.W} -= \text{learning\_rate} \times \text{layer.dW}$. |