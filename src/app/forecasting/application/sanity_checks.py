"""Shuffled-labels sanity check for direction classification pipelines (Ojala and Garriga, 2010)."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.forecasting.domain.protocols import IDirectionClassifier
from src.app.forecasting.domain.value_objects import (
    DirectionForecast,
    ShuffledLabelResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DA_LOWER_BOUND: float = 0.48
"""Lower bound for acceptable mean DA on shuffled labels."""

_DA_UPPER_BOUND: float = 0.52
"""Upper bound for acceptable mean DA on shuffled labels."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ShuffledLabelCheckConfig(BaseModel, frozen=True):
    """Configuration for the shuffled-labels permutation test.

    Attributes:
        n_permutations: Number of independent label-permutation rounds.
        random_seed: Base seed for the random number generator.
    """

    n_permutations: Annotated[
        int,
        PydanticField(ge=1, description="Number of label-permutation rounds"),
    ] = 5

    random_seed: int = 42
    """Base seed for the random number generator."""


# ---------------------------------------------------------------------------
# Shuffled-labels sanity check
# ---------------------------------------------------------------------------


def _evaluate_single_permutation(
    model_factory: type[IDirectionClassifier],
    model_kwargs: dict[str, object],
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    y_permuted: np.ndarray[tuple[int], np.dtype[np.float64]],
    folds: list[tuple[np.ndarray[tuple[int], np.dtype[np.intp]], np.ndarray[tuple[int], np.dtype[np.intp]]]],
) -> float:
    """Train on permuted labels across all folds, return mean DA.

    Args:
        model_factory: Classifier class to instantiate.
        model_kwargs: Keyword arguments passed to the model constructor.
        x: Full feature matrix of shape ``(n_samples, n_features)``.
        y_permuted: Permuted label vector of shape ``(n_samples,)``.
        folds: List of ``(train_indices, test_indices)`` tuples.

    Returns:
        Mean directional accuracy across all folds.
    """
    fold_das: list[float] = []
    for train_idx, test_idx in folds:
        model: IDirectionClassifier = model_factory(**model_kwargs)  # type: ignore[call-arg]
        model.fit(x[train_idx], y_permuted[train_idx])
        forecasts: list[DirectionForecast] = model.predict(x[test_idx])

        preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
            [f.predicted_direction for f in forecasts],
            dtype=np.float64,
        )
        fold_das.append(float(np.mean(preds == y_permuted[test_idx])))

    return float(np.mean(fold_das))


def run_shuffled_labels_check(  # noqa: PLR0913
    model_factory: type[IDirectionClassifier],
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    y: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    model_kwargs: dict[str, object],
    folds: list[tuple[np.ndarray[tuple[int], np.dtype[np.intp]], np.ndarray[tuple[int], np.dtype[np.intp]]]],
    config: ShuffledLabelCheckConfig | None = None,
) -> ShuffledLabelResult:
    """Run the shuffled-labels permutation test on any direction classifier.

    For each permutation round, labels are randomly permuted (destroying
    any real signal), a fresh model instance is trained on the permuted
    labels, and directional accuracy (DA) is evaluated on each fold's
    test set.  The per-fold DAs are averaged to produce a single DA for
    that permutation round.

    If the pipeline is correct (no data leakage), the mean DA across all
    permutation rounds should be approximately 50% (within +/-2pp).

    Reference:
        Ojala, M. and Garriga, G.C., 2010.  *Permutation tests for studying
        classifier performance.*  JMLR, 11, pp.1833-1863.

    Args:
        model_factory: Classifier class to instantiate (must accept ``**model_kwargs``).
        model_kwargs: Keyword arguments passed to the model constructor.
        x: Full feature matrix of shape ``(n_samples, n_features)``.
        y: Full label vector of shape ``(n_samples,)`` with values +1 or -1.
        folds: List of ``(train_indices, test_indices)`` tuples from CPCV or
            similar temporal splitting scheme.
        config: Optional configuration.  Defaults to ``ShuffledLabelCheckConfig()``.

    Returns:
        ShuffledLabelResult with per-permutation DA, mean DA, and pass/fail.

    Raises:
        ValueError: If ``x`` is empty, ``folds`` is empty, or ``n_permutations < 1``.
    """
    if x.shape[0] == 0:
        msg: str = "x must contain at least one sample"
        raise ValueError(msg)
    if len(folds) == 0:
        msg = "folds must not be empty"
        raise ValueError(msg)

    cfg: ShuffledLabelCheckConfig = config or ShuffledLabelCheckConfig()
    rng: np.random.Generator = np.random.default_rng(cfg.random_seed)
    model_name: str = model_factory.__name__

    per_permutation_da: list[float] = []
    for perm_idx in range(cfg.n_permutations):
        y_permuted: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.permutation(y).astype(np.float64)
        perm_da: float = _evaluate_single_permutation(model_factory, model_kwargs, x, y_permuted, folds)
        per_permutation_da.append(perm_da)

        logger.debug(
            "Shuffled-labels {} perm {}/{}: mean_DA={:.4f}",
            model_name,
            perm_idx + 1,
            cfg.n_permutations,
            perm_da,
        )

    mean_da: float = float(np.mean(per_permutation_da))
    passed: bool = _DA_LOWER_BOUND <= mean_da <= _DA_UPPER_BOUND

    logger.info(
        "Shuffled-labels check for {}: mean_DA={:.4f} (range [{:.4f}, {:.4f}]) → {}",
        model_name,
        mean_da,
        min(per_permutation_da),
        max(per_permutation_da),
        "PASSED" if passed else "FAILED",
    )

    return ShuffledLabelResult(
        model_name=model_name,
        n_permutations=cfg.n_permutations,
        per_permutation_da=tuple(per_permutation_da),
        mean_da=mean_da,
        passed=passed,
    )
