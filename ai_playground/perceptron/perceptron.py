import logging
import numpy as np

logger = logging.getLogger(__name__)


class Perceptron:
    """
    Single-layer perceptron using the Rosenblatt update rule.

    Learns a binary linear classifier via the perceptron learning rule:
        w += lr * (y - y_hat) * x
        b += lr * (y - y_hat)

    Converges when data is linearly separable; otherwise stops after max_epochs.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_epochs: int = 1000,
        bipolar: bool = False,
    ):
        """
        Parameters
        ----------
        bipolar : bool
            False (default) — labels are 0 / 1,  step fires at z >= 0.
            True            — labels are -1 / +1, step fires at z >= 0.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.bipolar = bipolar
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def _step(self, z: np.ndarray) -> np.ndarray:
        if self.bipolar:
            return np.where(z >= 0, 1, -1)
        return np.where(z >= 0, 1, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (0 or 1) for samples in X.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        labels : (n_samples,) — 0/1 when bipolar=False, -1/+1 when bipolar=True
        """
        if self.weights is None:
            raise RuntimeError("Call train() before predict().")
        return self._step(X @ self.weights + self.bias)

    def train(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Fit on labelled data.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,) — 0/1 when bipolar=False, -1/+1 when bipolar=True

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                z = xi @ self.weights + self.bias
                y_hat = self._step(z)
                delta = self.learning_rate * (yi - y_hat)
                logger.debug("Sample: %s, Label: %s, Weights: %s, Bias: %s z: %s", xi, yi, self.weights, self.bias, z)
                self.weights += delta * xi
                self.bias += delta
                errors += int(delta != 0)
            logger.debug("Weights after epoch %d: %s, bias: %.4f, errors: %d", epoch + 1, self.weights, self.bias, errors)
            if errors == 0:
                logger.info("Perceptron converged after %d epochs.", epoch + 1)
                break  # converged
        else:
            logger.warning(
                "Perceptron did not converge after %d epochs. "
                "Data may not be linearly separable.",
                self.max_epochs,
            )

        return self
