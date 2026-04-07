"""GRU classifier with MC Dropout for direction prediction (negative-result experiment)."""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.app.forecasting.domain.value_objects import (
    DirectionForecast,
    ForecastHorizon,
    GRUClassifierConfig,
)

# Validation split fraction for early stopping (last 20 % of training data).
_VAL_FRACTION: float = 0.20

# Decision threshold for binary direction: P(up) >= threshold → long (+1).
_DIRECTION_THRESHOLD: float = 0.5


class _GRUClassifierNetwork(nn.Module):
    """Multi-layer GRU with a sigmoid head for binary classification.

    Architecture: ``GRU(input_size, hidden_size, num_layers, dropout)``
    followed by ``Dropout`` and ``Linear(hidden_size, 1)`` with sigmoid
    activation, outputting P(direction = +1).

    Attributes:
        gru: Stacked GRU cell.
        dropout: Dropout layer for MC Dropout at inference.
        head: Linear projection from hidden state to logit.
        sigmoid: Sigmoid activation producing P(up).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialise the GRU classifier network architecture.

        Args:
            input_size: Number of input features per time step.
            hidden_size: Dimension of the GRU hidden state.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability between GRU layers and before the head.
        """
        super().__init__()
        self.gru: nn.GRU = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.head: nn.Linear = nn.Linear(hidden_size, 1)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRU + sigmoid head.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Probability predictions of shape ``(batch,)`` in [0, 1].
        """
        # output shape: (batch, seq_len, hidden_size)
        output: torch.Tensor
        output, _ = self.gru(x)
        # Take the last time-step hidden state
        last_hidden: torch.Tensor = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logit: torch.Tensor = self.head(last_hidden).squeeze(-1)
        prob: torch.Tensor = self.sigmoid(logit)
        return prob


class GRUClassifier:
    """GRU direction classifier with MC Dropout uncertainty estimation.

    Trains a multi-layer GRU on sliding-window sequences for binary
    direction prediction (+1 / -1).  Labels are converted to {0, 1}
    internally for BCE loss.  At inference, Monte Carlo Dropout produces
    epistemic uncertainty estimates by running multiple stochastic forward
    passes with dropout kept active.

    This model is part of a controlled negative-result experiment:
    Grinsztajn et al. (2022) show gradient-boosted trees outperform DL
    on tabular data below ~10K samples.  With ~5K dollar bars per asset,
    the GRU classifier is expected to underperform LightGBM.

    Attributes:
        config: GRU classifier configuration object.
        horizon: Forecast horizon embedded in every ``DirectionForecast``.
    """

    def __init__(self, config: GRUClassifierConfig, horizon: ForecastHorizon) -> None:
        """Initialise the GRU classifier.

        Args:
            config: GRU classifier configuration (hidden size, layers, dropout, etc.).
            horizon: Forecast horizon to embed in predictions.
        """
        self.config: GRUClassifierConfig = config
        self.horizon: ForecastHorizon = horizon
        self._model: _GRUClassifierNetwork | None = None
        self._scaler: StandardScaler = StandardScaler()
        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train the GRU classifier with early stopping on a held-out validation split.

        The feature matrix is first standardised (per-feature) and then
        converted into overlapping sequences of length
        ``config.sequence_length``.  Labels are converted from {-1, +1}
        to {0, 1} for BCE loss.  The last 20 % of sequences (in temporal
        order) form the validation set for early stopping.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Direction labels of shape ``(n_samples,)`` with values +1 or -1.

        Raises:
            ValueError: If inputs are empty or there are fewer samples than
                ``sequence_length + 1``.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        seq_len: int = self.config.sequence_length
        if n_samples < seq_len + 1:
            msg = f"Need at least sequence_length + 1 = {seq_len + 1} samples, got {n_samples}"
            raise ValueError(msg)

        self._set_seeds()

        # Convert y from {-1, +1} to {0, 1} for BCE loss
        y_binary: np.ndarray[tuple[int], np.dtype[np.float64]] = ((y_train + 1.0) / 2.0).astype(np.float64)

        train_tensors: tuple[torch.Tensor, torch.Tensor]
        val_tensors: tuple[torch.Tensor, torch.Tensor]
        n_train: int
        n_val: int
        train_tensors, val_tensors, n_train, n_val = self._prepare_data(x_train, y_binary)

        self._model = _GRUClassifierNetwork(
            input_size=x_train.shape[1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self._device)

        best_val_loss: float
        final_epoch: int
        best_val_loss, final_epoch = self._run_training_loop(train_tensors, val_tensors)

        logger.info(
            "GRU classifier fitted: {} epochs | best_val_loss={:.6f} | seq_len={} | train_seqs={} val_seqs={}",
            final_epoch,
            best_val_loss,
            seq_len,
            n_train,
            n_val,
        )

    # ------------------------------------------------------------------
    # Inference (MC Dropout)
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[DirectionForecast]:
        """Generate direction forecasts via MC Dropout (epistemic uncertainty).

        Runs ``config.mc_samples`` forward passes with dropout active.
        The mean probability across MC samples determines the direction:
        +1 if P(up) >= 0.5, else -1.  Confidence is ``max(P(up), 1-P(up))``.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of direction forecasts, one per output sequence.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x_test`` is empty or has fewer rows than ``sequence_length``.
        """
        if self._model is None:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_test: int = x_test.shape[0]
        if n_test == 0:
            msg = "x_test must contain at least one sample"
            raise ValueError(msg)

        seq_len: int = self.config.sequence_length
        if n_test < seq_len:
            msg = f"x_test must have at least sequence_length={seq_len} rows, got {n_test}"
            raise ValueError(msg)

        x_scaled: np.ndarray[tuple[int, int], np.dtype[np.float64]] = self._scaler.transform(x_test).astype(np.float64)

        # Build sequences from test data — target is irrelevant, use zeros
        x_seq: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
        x_seq, _ = self._make_sequences(x_scaled, np.zeros(n_test, dtype=np.float64), seq_len)

        x_tensor: torch.Tensor = torch.from_numpy(x_seq).float().to(self._device)

        # MC Dropout: keep model in train mode for active dropout
        self._model.train()
        mc_predictions: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []

        with torch.no_grad():
            for _ in range(self.config.mc_samples):
                probs: torch.Tensor = self._model(x_tensor)
                mc_predictions.append(probs.cpu().numpy().astype(np.float64))

        # Stack: (mc_samples, n_output_samples)
        stacked: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.stack(mc_predictions, axis=0)
        mean_probs: np.ndarray[tuple[int], np.dtype[np.float64]] = np.mean(stacked, axis=0).astype(np.float64)

        # Convert probabilities to DirectionForecast objects
        forecasts: list[DirectionForecast] = []
        for i in range(mean_probs.shape[0]):
            p_up: float = float(mean_probs[i])
            direction: int = 1 if p_up >= _DIRECTION_THRESHOLD else -1
            confidence: float = max(p_up, 1.0 - p_up)

            forecast: DirectionForecast = DirectionForecast(
                predicted_direction=direction,
                confidence=confidence,
                horizon=self.horizon,
            )
            forecasts.append(forecast)

        return forecasts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_binary: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], int, int]:
        """Standardise features, build sequences, and split into train/val tensors.

        Args:
            x_train: Raw feature matrix ``(n_samples, n_features)``.
            y_binary: Binary target vector ``(n_samples,)`` with values in {0, 1}.

        Returns:
            Tuple of ``(train_tensors, val_tensors, n_train, n_val)`` where each
            tensor pair is ``(x_sequences, y_targets)`` on the target device.
        """
        seq_len: int = self.config.sequence_length

        x_scaled: np.ndarray[tuple[int, int], np.dtype[np.float64]] = self._scaler.fit_transform(x_train).astype(
            np.float64
        )

        x_seq: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
        y_seq: np.ndarray[tuple[int], np.dtype[np.float64]]
        x_seq, y_seq = self._make_sequences(x_scaled, y_binary, seq_len)

        n_seq: int = x_seq.shape[0]
        n_val: int = max(1, int(n_seq * _VAL_FRACTION))
        n_train: int = n_seq - n_val

        x_train_t: torch.Tensor = torch.from_numpy(x_seq[:n_train]).float().to(self._device)
        y_train_t: torch.Tensor = torch.from_numpy(y_seq[:n_train]).float().to(self._device)
        x_val_t: torch.Tensor = torch.from_numpy(x_seq[n_train:]).float().to(self._device)
        y_val_t: torch.Tensor = torch.from_numpy(y_seq[n_train:]).float().to(self._device)

        return (x_train_t, y_train_t), (x_val_t, y_val_t), n_train, n_val

    def _run_training_loop(
        self,
        train_tensors: tuple[torch.Tensor, torch.Tensor],
        val_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[float, int]:
        """Execute the training loop with early stopping.

        Args:
            train_tensors: ``(x_train_seq, y_train_seq)`` on device.
            val_tensors: ``(x_val_seq, y_val_seq)`` on device.

        Returns:
            Tuple of ``(best_val_loss, final_epoch_number)``.
        """
        assert self._model is not None  # noqa: S101

        x_train_seq: torch.Tensor
        y_train_seq: torch.Tensor
        x_train_seq, y_train_seq = train_tensors
        x_val_seq: torch.Tensor
        y_val_seq: torch.Tensor
        x_val_seq, y_val_seq = val_tensors

        optimizer: torch.optim.Adam = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
        )
        loss_fn: nn.BCELoss = nn.BCELoss()

        best_val_loss: float = float("inf")
        patience_counter: int = 0
        best_state: dict[str, torch.Tensor] = {}
        epoch: int = 0

        for epoch in range(self.config.n_epochs):
            # --- Training step ---
            self._model.train()
            train_loss: float = self._train_epoch(
                x_train_seq,
                y_train_seq,
                optimizer,
                loss_fn,
            )

            # --- Validation step ---
            self._model.eval()
            with torch.no_grad():
                val_preds: torch.Tensor = self._model(x_val_seq)
                val_loss: float = loss_fn(val_preds, y_val_seq).item()

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.info(
                    "Early stopping at epoch {} | train_loss={:.6f} val_loss={:.6f}",
                    epoch + 1,
                    train_loss,
                    val_loss,
                )
                break

        # Restore best weights
        if best_state:
            self._model.load_state_dict(best_state)

        return best_val_loss, epoch + 1

    def _set_seeds(self) -> None:
        """Set random seeds for NumPy, Python, and PyTorch reproducibility."""
        seed: int = self.config.random_seed
        np.random.seed(seed)  # noqa: NPY002
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _train_epoch(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        optimizer: torch.optim.Adam,
        loss_fn: nn.BCELoss,
    ) -> float:
        """Run one training epoch with mini-batch gradient descent.

        Args:
            x_train: Training sequences ``(n_train, seq_len, n_features)``.
            y_train: Training targets ``(n_train,)`` with values in {0, 1}.
            optimizer: Adam optimiser.
            loss_fn: BCE loss function.

        Returns:
            Mean training loss for the epoch.
        """
        assert self._model is not None  # noqa: S101

        n_samples: int = x_train.shape[0]
        batch_size: int = self.config.batch_size
        indices: torch.Tensor = torch.randperm(n_samples, device=self._device)

        total_loss: float = 0.0
        n_batches: int = 0

        for start in range(0, n_samples, batch_size):
            end: int = min(start + batch_size, n_samples)
            batch_idx: torch.Tensor = indices[start:end]

            x_batch: torch.Tensor = x_train[batch_idx]
            y_batch: torch.Tensor = y_train[batch_idx]

            optimizer.zero_grad()
            preds: torch.Tensor = self._model(x_batch)
            loss: torch.Tensor = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        mean_loss: float = total_loss / max(n_batches, 1)
        return mean_loss

    @staticmethod
    def _make_sequences(
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y: np.ndarray[tuple[int], np.dtype[np.float64]],
        seq_len: int,
    ) -> tuple[
        np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int], np.dtype[np.float64]],
    ]:
        """Convert flat feature matrix into overlapping sequences.

        Each sequence is a sliding window of ``seq_len`` consecutive rows.
        The target for each sequence is the value at the last time step.

        Args:
            x: Scaled feature matrix ``(n_samples, n_features)``.
            y: Target vector ``(n_samples,)``.
            seq_len: Number of time steps per sequence.

        Returns:
            Tuple of (sequences, targets) where sequences has shape
            ``(n_sequences, seq_len, n_features)`` and targets has shape
            ``(n_sequences,)``.
        """
        n_samples: int = x.shape[0]
        n_features: int = x.shape[1]
        n_sequences: int = n_samples - seq_len

        x_seq: np.ndarray[tuple[int, int, int], np.dtype[np.float64]] = np.empty(
            (n_sequences, seq_len, n_features), dtype=np.float64
        )
        y_seq: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(n_sequences, dtype=np.float64)

        for i in range(n_sequences):
            x_seq[i] = x[i : i + seq_len]
            y_seq[i] = y[i + seq_len]

        return x_seq, y_seq
