import logging

import numpy as np

from .activations import Activation
from .losses import Loss

logger = logging.getLogger(__name__)


class MLP:
    """Multi-layer perceptron with fully-connected layers.

    Architecture
    ------------
    Input → [Linear → hidden_activation] × (L-1) → Linear → output_activation

    Convention
    ----------
    Weights follow the z = Wa + b convention:
      W⁽ˡ⁾ : (n_out, n_in)
      b⁽ˡ⁾ : (n_out,)
      z⁽ˡ⁾ : (n_out, n_samples)
      a⁽ˡ⁾ : (n_out, n_samples)

    Data is accepted and returned in row-major form (n_samples, n_features)
    and transposed internally.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        hidden_activation: Activation,
        output_activation: Activation,
        loss: Loss,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        layer_sizes : [n_input, h₁, h₂, …, n_output]
            Number of units in the input, each hidden, and the output layer.
        hidden_activation : Activation
            Activation applied after every hidden layer.
        output_activation : Activation
            Activation applied after the final layer.
        loss : Loss
            Loss function used to compute the output delta during backprop.
        seed : int or None
            Seed for the weight initialisation RNG. Pass an integer for
            reproducible results; None (default) draws from OS entropy.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 entries (input and output).")

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss

        # He initialisation: W⁽ˡ⁾ ~ N(0, 2/n_in), shape (n_out, n_in)
        self._rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(self._rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in))
            self.biases.append(np.zeros(n_out))

        # Populated by forward(); consumed by backward().
        self._zs: list[np.ndarray] = []   # pre-activations z⁽ˡ⁾, each (n_out, n_samples)
        self._as: list[np.ndarray] = []   # activations    a⁽ˡ⁾, each (n_out, n_samples); a⁽⁰⁾ = Xᵀ

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the forward pass, caching pre-activations and activations.

        Parameters
        ----------
        X : (n_samples, n_input)

        Returns
        -------
        output : (n_samples, n_output) — activations of the final layer
        """
        self._zs = []
        self._as = [X.T]          # a⁽⁰⁾ = Xᵀ, shape (n_input, n_samples)

        n_layers = len(self.weights)
        a = X.T
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b[:, np.newaxis]    # (n_out, n_samples)
            self._zs.append(z)
            activation = self.output_activation if l == n_layers - 1 else self.hidden_activation
            a = activation.forward(z)
            self._as.append(a)

        return a.T                # return (n_samples, n_output)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute gradients via backpropagation.

        Must be called after forward().

        Parameters
        ----------
        y : (n_samples, n_output) ground-truth targets

        Returns
        -------
        dW : list of weight gradients ∂L/∂W⁽ˡ⁾, each (n_out, n_in)
        db : list of bias gradients  ∂L/∂b⁽ˡ⁾, each (n_out,)
        """
        n_layers = len(self.weights)
        dW = [None] * n_layers
        db = [None] * n_layers

        # Output delta: δ⁽ᴸ⁾ = dL/dŷ ⊙ σ'(z⁽ᴸ⁾),  shape (n_out, n_samples)
        y_pred = self._as[-1]                                       # (n_out, n_samples)
        n_samples = y_pred.shape[1]
        delta = (
            self.loss.gradient(y.T, y_pred)
            * self.output_activation.derivative(self._zs[-1])
        )

        for l in reversed(range(n_layers)):
            a_prev = self._as[l]                    # (n_in, n_samples)
            dW[l] = (delta @ a_prev.T) / n_samples  # (n_out, n_in)
            db[l] = delta.sum(axis=1) / n_samples   # (n_out,)

            if l > 0:
                # Propagate: δ⁽ˡ⁾ = W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)
                delta = (self.weights[l].T @ delta) * self.hidden_activation.derivative(self._zs[l - 1])

        return dW, db

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    def update(self, dW: list[np.ndarray], db: list[np.ndarray], lr: float) -> None:
        """Apply vanilla gradient descent.

        Parameters
        ----------
        dW : weight gradients from backward()
        db : bias gradients from backward()
        lr : learning rate
        """
        for l in range(len(self.weights)):
            self.weights[l] -= lr * dW[l]
            self.biases[l]  -= lr * db[l]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int | None = None,
        log_every: int = 100,
    ) -> list[float]:
        """Train the network with gradient descent.

        Parameters
        ----------
        X          : (n_samples, n_input) input data
        y          : (n_samples, n_output) targets
        epochs     : number of full passes over the data
        lr         : learning rate
        batch_size : number of samples per gradient update.
                     None  → full-batch GD (default)
                     1     → stochastic GD
                     k     → mini-batch GD
        log_every  : log the loss every this many epochs (0 = never)

        Returns
        -------
        losses : mean loss over each epoch's batches, one value per epoch
        """
        n_samples = X.shape[0]
        effective_batch = n_samples if batch_size is None else batch_size

        losses = []
        for epoch in range(1, epochs + 1):
            indices = self._rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, effective_batch):
                batch_idx = indices[start : start + effective_batch]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                y_pred = self.forward(X_batch)
                epoch_loss += self.loss.compute_loss(y_batch.T, y_pred.T)
                n_batches += 1

                dW, db = self.backward(y_batch)
                self.update(dW, db, lr)

            loss_val = epoch_loss / n_batches
            losses.append(loss_val)

            if log_every and epoch % log_every == 0:
                logger.info("Epoch %d/%d  loss=%.6f", epoch, epochs, loss_val)

        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run a forward pass without updating the cache (inference only).

        Parameters
        ----------
        X : (n_samples, n_input)

        Returns
        -------
        output : (n_samples, n_output)
        """
        return self.forward(X)
