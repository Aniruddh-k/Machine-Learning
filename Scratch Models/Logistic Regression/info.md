# Logistic Regression Implementation

A simple and efficient implementation of Logistic Regression for binary classification using gradient descent optimization.

## Overview

Logistic Regression is a statistical method used for binary classification problems. Unlike linear regression, it uses the logistic function (sigmoid) to map any real-valued input to a value between 0 and 1, making it suitable for probability estimation and classification tasks.

## Mathematical Foundation

### 1. Sigmoid Function

The sigmoid function maps any real number to a value between 0 and 1:

```
σ(z) = 1 / (1 + e^(-z))
```

Where:
- `z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = X·w + b`
- `w` is the weight vector
- `b` is the bias term
- `X` is the feature matrix

### 2. Prediction Formula

For a given input `X`, the probability of belonging to class 1 is:

```
P(y=1|X) = σ(X·w + b) = 1 / (1 + e^(-(X·w + b)))
```

### 3. Binary Cross-Entropy Loss

The cost function used in logistic regression is the binary cross-entropy:

```
J(w,b) = -1/m ∑[y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-ŷ⁽ⁱ⁾)]
```

Where:
- `m` is the number of training examples
- `y⁽ⁱ⁾` is the actual label for the i-th example
- `ŷ⁽ⁱ⁾` is the predicted probability for the i-th example

### 4. Gradient Descent Optimization

To minimize the cost function, we use gradient descent. The gradients are:

#### Gradient with respect to weights (w):
```
∂J/∂w = 1/m × X^T × (ŷ - y)
```

#### Gradient with respect to bias (b):
```
∂J/∂b = 1/m × ∑(ŷ - y)
```

#### Weight Update Rules:
```
w := w - α × ∂J/∂w
b := b - α × ∂J/∂b
```

Where `α` is the learning rate.

## Class Documentation

### `Logistic_Regression`

A class implementing logistic regression with gradient descent optimization.

#### Parameters

- **learning_rate** (float): The step size for gradient descent updates. Controls how much the weights are adjusted during each iteration.
- **no_of_iterations** (int): The number of iterations to run the gradient descent algorithm.

#### Methods

##### `__init__(self, learning_rate, no_of_iterations)`
Initialize the logistic regression model with specified hyperparameters.

##### `fit(self, X, Y)`
Train the logistic regression model on the given dataset.

**Parameters:**
- `X` (numpy.ndarray): Feature matrix of shape (m, n) where m is the number of samples and n is the number of features.
- `Y` (numpy.ndarray): Target vector of shape (m,) containing binary labels (0 or 1).

**Process:**
1. Initialize weights to zero and bias to zero
2. For each iteration, update weights using gradient descent
3. Store the optimized weights and bias

##### `update_weights(self)`
Internal method to update weights and bias using gradient descent.

**Steps:**
1. Calculate linear combination: `z = X·w + b`
2. Apply sigmoid function: `ŷ = σ(z)`
3. Compute gradients for weights and bias
4. Update weights and bias using learning rate

##### `predict(self, X)`
Make predictions on new data.

**Parameters:**
- `X` (numpy.ndarray): Feature matrix of shape (m, n) for prediction.

**Returns:**
- `list`: Binary predictions (0 or 1) based on 0.5 threshold.

**Process:**
1. Calculate probabilities using trained weights
2. Apply threshold (0.5) to convert probabilities to binary predictions

## Usage Example

```python
import numpy as np
from logistic_regression import Logistic_Regression

# Generate sample data
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create and train model
model = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Calculate accuracy
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

## Key Features

- **Simple Implementation**: Easy to understand and modify
- **Vectorized Operations**: Efficient computation using NumPy
- **Binary Classification**: Designed for binary classification problems
- **Gradient Descent**: Uses standard gradient descent optimization
- **Probability Output**: Can output probabilities before thresholding

## Dependencies

- NumPy: For numerical computations and array operations

## Mathematical Intuition

The logistic regression model learns a decision boundary that separates the two classes. The sigmoid function ensures that predictions are always between 0 and 1, making them interpretable as probabilities. The binary cross-entropy loss function is convex, guaranteeing that gradient descent will find the global minimum.

The algorithm iteratively adjusts the weights to minimize the difference between predicted probabilities and actual labels, effectively learning the optimal decision boundary for the given data.

## Notes

- The threshold for classification is set to 0.5. Adjust this based on your specific use case.
- Consider feature scaling for better convergence, especially with features of different scales.
- Monitor the loss function to ensure proper convergence.
- For large datasets, consider using stochastic or mini-batch gradient descent for better performance.
