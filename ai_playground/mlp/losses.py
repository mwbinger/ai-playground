from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class LossType(Enum):
    SQUARED_ERROR = "squared_error"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    CATEGORICAL_CROSS_ENTROPY = "categorical_cross_entropy"


class Loss(ABC):
    """Abstract interface for loss functions used by the MLP.

    Each implementation provides a scalar loss and its gradient with respect
    to the network's output (y_pred), which seeds backpropagation.
    """

    @property
    @abstractmethod
    def name(self) -> LossType:
        """Return the loss identifier used by this implementation."""

    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return the total loss summed over the batch.

        Parameters
        ----------
        y_true : (n_samples, ...) ground-truth targets
        y_pred : (n_samples, ...) network outputs

        Returns
        -------
        scalar summed loss
        """

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Return dL/dy_pred, the gradient of the loss w.r.t. network outputs.

        Parameters
        ----------
        y_true : (n_samples, ...) ground-truth targets
        y_pred : (n_samples, ...) network outputs

        Returns
        -------
        gradient array with the same shape as y_pred
        """


class SquaredError(Loss):
    """Summed squared error: L = (1/2) * sum((y_pred - y_true)^2).

    The 1/2 factor cancels the 2 from the derivative, giving a clean gradient.
    """

    @property
    def name(self) -> LossType:
        return LossType.SQUARED_ERROR

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(0.5 * np.sum((y_pred - y_true) ** 2))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true


class BinaryCrossEntropy(Loss):
    """Summed binary cross-entropy: L = -sum(y*log(p) + (1-y)*log(1-p)).

    Expects y_pred to be probabilities in (0, 1) — i.e. sigmoid output.
    """

    _EPS = 1e-12  # clip to avoid log(0)

    @property
    def name(self) -> LossType:
        return LossType.BINARY_CROSS_ENTROPY

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, self._EPS, 1.0 - self._EPS)
        return float(-np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, self._EPS, 1.0 - self._EPS)
        return -(y_true / p) + (1.0 - y_true) / (1.0 - p)


class CategoricalCrossEntropy(Loss):
    """Summed categorical cross-entropy: L = -sum(sum(y * log(p), axis=-1)).

    Expects y_true as one-hot vectors and y_pred as softmax probabilities.
    """

    _EPS = 1e-12

    @property
    def name(self) -> LossType:
        return LossType.CATEGORICAL_CROSS_ENTROPY

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, self._EPS, 1.0)
        return float(-np.sum(y_true * np.log(p)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, self._EPS, 1.0)
        return -(y_true / p)