"""Unit tests for ingestion domain exceptions.

Verifies that the exception hierarchy is correctly structured and that
exceptions carry messages as expected.
"""

from __future__ import annotations

import pytest

from src.app.ingestion.domain.exceptions import FetchError, IngestionError, RateLimitError


class TestIngestionError:
    """Tests for the ``IngestionError`` base exception."""

    def test_is_exception_subclass(self) -> None:
        """IngestionError must inherit from the built-in Exception class."""
        assert issubclass(IngestionError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """IngestionError must be raisable and catchable as itself."""
        with pytest.raises(IngestionError):
            raise IngestionError("base ingestion error")

    def test_carries_message_correctly(self) -> None:
        """IngestionError must expose the message via args[0]."""
        msg: str = "something went wrong in ingestion"
        exc: IngestionError = IngestionError(msg)
        assert exc.args[0] == msg


class TestFetchError:
    """Tests for the ``FetchError`` exception."""

    def test_is_subclass_of_ingestion_error(self) -> None:
        """FetchError must be a subclass of IngestionError."""
        assert issubclass(FetchError, IngestionError)

    def test_is_subclass_of_exception(self) -> None:
        """FetchError must be a subclass of the built-in Exception class."""
        assert issubclass(FetchError, Exception)

    def test_can_be_raised_and_caught_as_ingestion_error(self) -> None:
        """FetchError must be catchable as IngestionError due to inheritance."""
        with pytest.raises(IngestionError):
            raise FetchError("all retries exhausted")

    def test_can_be_raised_and_caught_as_fetch_error(self) -> None:
        """FetchError must be catchable as its own type."""
        with pytest.raises(FetchError):
            raise FetchError("all retries exhausted")

    def test_carries_message_correctly(self) -> None:
        """FetchError must preserve the error message in args[0]."""
        msg: str = "all 5 retries exhausted for BTCUSDT"
        exc: FetchError = FetchError(msg)
        assert exc.args[0] == msg

    def test_chained_exception_preserved(self) -> None:
        """FetchError must support exception chaining via ``from exc`` syntax."""
        cause: RuntimeError = RuntimeError("network timeout")
        try:
            raise FetchError("fetch failed") from cause
        except FetchError as err:
            assert err.__cause__ is cause


class TestRateLimitError:
    """Tests for the ``RateLimitError`` exception."""

    def test_is_subclass_of_ingestion_error(self) -> None:
        """RateLimitError must be a subclass of IngestionError."""
        assert issubclass(RateLimitError, IngestionError)

    def test_is_subclass_of_exception(self) -> None:
        """RateLimitError must be a subclass of the built-in Exception class."""
        assert issubclass(RateLimitError, Exception)

    def test_can_be_raised_and_caught_as_ingestion_error(self) -> None:
        """RateLimitError must be catchable as IngestionError due to inheritance."""
        with pytest.raises(IngestionError):
            raise RateLimitError("rate limit hit")

    def test_can_be_raised_and_caught_as_rate_limit_error(self) -> None:
        """RateLimitError must be catchable as its own type."""
        with pytest.raises(RateLimitError):
            raise RateLimitError("Binance rate limit exhausted for BTCUSDT")

    def test_carries_message_correctly(self) -> None:
        """RateLimitError must preserve the error message in args[0]."""
        msg: str = "Binance rate limit hit for BTCUSDT"
        exc: RateLimitError = RateLimitError(msg)
        assert exc.args[0] == msg

    def test_chained_exception_preserved(self) -> None:
        """RateLimitError must support exception chaining via ``from exc`` syntax."""
        cause: RuntimeError = RuntimeError("HTTP 429")
        try:
            raise RateLimitError("rate limit") from cause
        except RateLimitError as err:
            assert err.__cause__ is cause

    def test_is_not_subclass_of_fetch_error(self) -> None:
        """RateLimitError must NOT inherit from FetchError — they are siblings."""
        assert not issubclass(RateLimitError, FetchError)

    def test_fetch_error_is_not_subclass_of_rate_limit_error(self) -> None:
        """FetchError must NOT inherit from RateLimitError — they are siblings."""
        assert not issubclass(FetchError, RateLimitError)
