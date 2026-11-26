
import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:

    # Convert inputs into NumPy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Number of samples and features
    n_samples, n_features = X.shape

    # Initialize weights and bias
    w = np.zeros(n_features)
    b = 0.0

    # Training loop
    for _ in range(steps):

        # 1. Forward propagation
        z = X @ w + b
        y_pred = _sigmoid(z)

        # 2. Calculate gradients
        error = y_pred - y

        dw = (X.T @ error) / n_samples
        db = np.mean(error)

        # 3. Update weights and bias
        w -= lr * dw
        b -= lr * db

    return w, b
