"""GRU regressor with MC Dropout for return magnitude prediction."""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.app.forecasting.domain.value_objects import GRUConfig, PointPrediction

# Validation split fraction for early stopping (last 20 % of training data).
_VAL_FRACTION: float = 0.20


class _GRUNetwork(nn.Module):
    """Multi-layer GRU with a linear head for scalar regression.

    Architecture: ``GRU(input_size, hidden_size, num_layers, dropout)``
    followed by ``Linear(hidden_size, 1)`` applied to the last time-step
    hidden state.

    Attributes:
        gru: Stacked GRU cell.
        head: Linear projection from hidden state to scalar output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialise the GRU network architecture.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRU + linear head.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Scalar predictions of shape ``(batch,)``.
        """
        # output shape: (batch, seq_len, hidden_size)
        output: torch.Tensor
        output, _ = self.gru(x)
        # Take the last time-step hidden state
        last_hidden: torch.Tensor = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        prediction: torch.Tensor = self.head(last_hidden).squeeze(-1)
        return prediction


class GRURegressor:
    """GRU regressor with MC Dropout uncertainty estimation.

    Trains a multi-layer GRU on sliding-window sequences extracted from
    the feature matrix.  At inference, Monte Carlo Dropout produces
    epistemic uncertainty estimates by running multiple stochastic forward
    passes with dropout kept active.

    Attributes:
        config: GRU configuration object.
    """

    def __init__(self, config: GRUConfig) -> None:
        """Initialise the GRU regressor.

        Args:
            config: GRU configuration (hidden size, layers, dropout, seed, etc.).
        """
        self.config: GRUConfig = config
        self._model: _GRUNetwork | None = None
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
        """Train the GRU model with early stopping on a held-out validation split.

        The feature matrix is first standardised (per-feature) and then
        converted into overlapping sequences of length
        ``config.sequence_length``.  The last 20 % of sequences (in
        temporal order) form the validation set for early stopping.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Target vector of shape ``(n_samples,)``.

        Raises:
            ValueError: If there are fewer samples than ``sequence_length + 1``.
        """
        seq_len: int = self.config.sequence_length
        if x_train.shape[0] < seq_len + 1:
            msg: str = f"Need at least sequence_length + 1 = {seq_len + 1} samples, got {x_train.shape[0]}"
            raise ValueError(msg)

        self._set_seeds()

        train_tensors, val_tensors, n_train, n_val = self._prepare_data(x_train, y_train)

        self._model = _GRUNetwork(
            input_size=x_train.shape[1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self._device)

        best_val_loss: float
        final_epoch: int
        best_val_loss, final_epoch = self._run_training_loop(train_tensors, val_tensors)

        logger.info(
            "GRU fitted: {} epochs | best_val_loss={:.6f} | seq_len={} | train_seqs={} val_seqs={}",
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
    ) -> PointPrediction:
        """Generate predictions via MC Dropout (epistemic uncertainty).

        Runs ``config.mc_samples`` forward passes with dropout active.
        The mean of all passes is the point estimate; the standard
        deviation captures model uncertainty.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Point prediction with MC Dropout mean and std.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x_test`` has fewer rows than ``sequence_length``.
        """
        if self._model is None:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        seq_len: int = self.config.sequence_length
        if x_test.shape[0] < seq_len:
            msg = f"x_test must have at least sequence_length={seq_len} rows, got {x_test.shape[0]}"
            raise ValueError(msg)

        x_scaled: np.ndarray[tuple[int, int], np.dtype[np.float64]] = self._scaler.transform(x_test).astype(np.float64)

        # Build sequences from test data — target is irrelevant, use zeros
        dummy_y: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(x_test.shape[0], dtype=np.float64)
        x_seq: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
        x_seq, _ = self._make_sequences(x_scaled, dummy_y, seq_len)

        x_tensor: torch.Tensor = torch.from_numpy(x_seq).float().to(self._device)

        # MC Dropout: keep model in train mode for active dropout
        self._model.train()
        mc_predictions: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []

        with torch.no_grad():
            for _ in range(self.config.mc_samples):
                preds: torch.Tensor = self._model(x_tensor)
                mc_predictions.append(preds.cpu().numpy().astype(np.float64))

        # Stack: (mc_samples, n_output_samples)
        stacked: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.stack(mc_predictions, axis=0)
        mean: np.ndarray[tuple[int], np.dtype[np.float64]] = np.mean(stacked, axis=0).astype(np.float64)
        std: np.ndarray[tuple[int], np.dtype[np.float64]] = np.std(stacked, axis=0).astype(np.float64)

        return PointPrediction(mean=mean, std=std)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], int, int]:
        """Standardise features, build sequences, and split into train/val tensors.

        Args:
            x_train: Raw feature matrix ``(n_samples, n_features)``.
            y_train: Target vector ``(n_samples,)``.

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
        x_seq, y_seq = self._make_sequences(x_scaled, y_train, seq_len)

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

        x_train_seq, y_train_seq = train_tensors
        x_val_seq, y_val_seq = val_tensors

        optimizer: torch.optim.Adam = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
        )
        loss_fn: nn.MSELoss = nn.MSELoss()

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
        loss_fn: nn.MSELoss,
    ) -> float:
        """Run one training epoch with mini-batch gradient descent.

        Args:
            x_train: Training sequences ``(n_train, seq_len, n_features)``.
            y_train: Training targets ``(n_train,)``.
            optimizer: Adam optimiser.
            loss_fn: MSE loss function.

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
