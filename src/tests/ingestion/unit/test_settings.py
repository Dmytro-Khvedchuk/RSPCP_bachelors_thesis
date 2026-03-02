"""Unit tests for ``BinanceSettings`` infrastructure configuration.

Settings are loaded from environment variables.  Each test uses
``monkeypatch.setenv()`` to inject required variables without polluting
the real environment or touching the ``.env`` file.

Note on missing-key tests: ``BinanceSettings`` reads from ``.env`` as a
fallback source, which would silently supply missing fields even after
``monkeypatch.delenv``.  Tests that need to verify required-field validation
use ``BinanceSettingsNoEnvFile`` from the ingestion conftest, which disables
``.env`` reading entirely.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.app.ingestion.infrastructure.settings import BinanceSettings
from src.tests.ingestion.conftest import BinanceSettingsNoEnvFile, FAKE_API_KEY, FAKE_SECRET_KEY


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES: int = 5
_DEFAULT_RETRY_MIN_WAIT: int = 1
_DEFAULT_RETRY_MAX_WAIT: int = 10
_DEFAULT_BATCH_SIZE: int = 1000


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBinanceSettings:
    """Tests for ``BinanceSettings`` pydantic-settings model."""

    def test_loads_api_key_from_environment(self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """api_key must be read from the BINANCE_API_KEY environment variable."""
        settings: BinanceSettings = BinanceSettings()  # type: ignore[call-arg]
        assert settings.api_key == FAKE_API_KEY

    def test_loads_secret_key_from_environment(self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """secret_key must be read from the BINANCE_SECRET_KEY environment variable."""
        settings: BinanceSettings = BinanceSettings()  # type: ignore[call-arg]
        assert settings.secret_key == FAKE_SECRET_KEY

    def test_default_max_retries(self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_retries must default to 5 when not explicitly set."""
        settings: BinanceSettings = BinanceSettings()  # type: ignore[call-arg]
        assert settings.max_retries == _DEFAULT_MAX_RETRIES

    def test_default_retry_min_wait(self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """retry_min_wait must default to 1 when not explicitly set."""
        settings: BinanceSettings = BinanceSettings()  # type: ignore[call-arg]
        assert settings.retry_min_wait == _DEFAULT_RETRY_MIN_WAIT

    def test_default_retry_max_wait(self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """retry_max_wait must default to 10 when not explicitly set."""
        settings: BinanceSettings = BinanceSettings()  # type: ignore[call-arg]
        assert settings.retry_max_wait == _DEFAULT_RETRY_MAX_WAIT

    def test_default_batch_size(self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """batch_size must default to 1000 when not explicitly set."""
        settings: BinanceSettings = BinanceSettings()  # type: ignore[call-arg]
        assert settings.batch_size == _DEFAULT_BATCH_SIZE

    def test_custom_max_retries_from_environment(self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_retries must be overridable via the BINANCE_MAX_RETRIES variable."""
        monkeypatch.setenv("BINANCE_MAX_RETRIES", "3")
        settings: BinanceSettings = BinanceSettings()  # type: ignore[call-arg]
        assert settings.max_retries == 3

    def test_custom_batch_size_from_environment(self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """batch_size must be overridable via the BINANCE_BATCH_SIZE variable."""
        monkeypatch.setenv("BINANCE_BATCH_SIZE", "500")
        settings: BinanceSettings = BinanceSettings()  # type: ignore[call-arg]
        assert settings.batch_size == 500

    def test_missing_api_key_raises_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting BINANCE_API_KEY must raise ValidationError for a required field.

        Uses ``BinanceSettingsNoEnvFile`` to ensure ``.env`` cannot silently
        supply the missing value.
        """
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        monkeypatch.setenv("BINANCE_SECRET_KEY", FAKE_SECRET_KEY)

        with pytest.raises(ValidationError):
            BinanceSettingsNoEnvFile()  # type: ignore[call-arg]

    def test_missing_secret_key_raises_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting BINANCE_SECRET_KEY must raise ValidationError for a required field.

        Uses ``BinanceSettingsNoEnvFile`` to ensure ``.env`` cannot silently
        supply the missing value.
        """
        monkeypatch.setenv("BINANCE_API_KEY", FAKE_API_KEY)
        monkeypatch.delenv("BINANCE_SECRET_KEY", raising=False)

        with pytest.raises(ValidationError):
            BinanceSettingsNoEnvFile()  # type: ignore[call-arg]

    def test_batch_size_above_1000_raises_validation_error(
        self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """batch_size > 1000 (Binance API limit) must raise a ValidationError."""
        monkeypatch.setenv("BINANCE_BATCH_SIZE", "1001")

        with pytest.raises(ValidationError):
            BinanceSettings()  # type: ignore[call-arg]

    def test_batch_size_zero_raises_validation_error(
        self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """batch_size = 0 must raise a ValidationError (ge=1 constraint)."""
        monkeypatch.setenv("BINANCE_BATCH_SIZE", "0")

        with pytest.raises(ValidationError):
            BinanceSettings()  # type: ignore[call-arg]

    def test_max_retries_zero_raises_validation_error(
        self, set_binance_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_retries = 0 must raise a ValidationError (ge=1 constraint)."""
        monkeypatch.setenv("BINANCE_MAX_RETRIES", "0")

        with pytest.raises(ValidationError):
            BinanceSettings()  # type: ignore[call-arg]

    def test_settings_can_be_model_constructed_without_env(self) -> None:
        """model_construct() must bypass env loading for use in tests."""
        settings: BinanceSettings = BinanceSettings.model_construct(
            api_key="direct_key",
            secret_key="direct_secret",
            max_retries=3,
            retry_min_wait=1,
            retry_max_wait=5,
            batch_size=500,
        )
        assert settings.api_key == "direct_key"
        assert settings.secret_key == "direct_secret"
        assert settings.max_retries == 3
        assert settings.batch_size == 500
