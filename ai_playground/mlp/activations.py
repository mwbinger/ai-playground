from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

class ActivationType(Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    RELU = "relu"
    TANH = "tanh"
    LINEAR = "linear"

class Activation(ABC):
    """Abstract interface for activation functions used by MLP layers.

    Implementations must provide a forward pass and the derivative of the
    activation with respect to the pre-activation input z.
    """

    @property
    @abstractmethod
    def name(self) -> ActivationType:
        """Return the activation identifier used by this implementation."""

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """Compute activation outputs for pre-activation inputs z.

        Parameters
        ----------
        z : (n_samples, n_features) pre-activation inputs

        Returns
        -------
        activations with the same shape as z
        """

    @abstractmethod
    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Compute d(activation)/dz for pre-activation inputs z.

        Parameters
        ----------
        z : (n_samples, n_features) pre-activation inputs

        Returns
        -------
        elementwise derivatives with the same shape as z
        """


class Sigmoid(Activation):
    @property
    def name(self) -> ActivationType:
        return ActivationType.SIGMOID

    def forward(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        s = self.forward(z)
        return s * (1.0 - s)


class Softmax(Activation):
    @property
    def name(self) -> ActivationType:
        return ActivationType.SOFTMAX

    def forward(self, z: np.ndarray) -> np.ndarray:
        # subtract max per row for numerical stability
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # Returns the diagonal of the Jacobian: s*(1-s)
        # (full Jacobian is s_i*(delta_ij - s_j); this is the elementwise form
        # used when the upstream gradient is already scaled by the loss derivative)
        s = self.forward(z)
        return s * (1.0 - s)


class ReLU(Activation):
    @property
    def name(self) -> ActivationType:
        return ActivationType.RELU

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)


class Tanh(Activation):
    @property
    def name(self) -> ActivationType:
        return ActivationType.TANH

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(z) ** 2


class Linear(Activation):
    @property
    def name(self) -> ActivationType:
        return ActivationType.LINEAR

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)
