"""Unit tests for the ``IngestionService`` application service.

All external dependencies (``IMarketDataFetcher``, ``IOHLCVRepository``) are
replaced with in-memory Protocol-compatible fakes so that only the service
logic is exercised.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

from src.app.ingestion.application.commands import IngestAssetCommand, IngestUniverseCommand
from src.app.ingestion.application.services import IngestionService
from src.app.ingestion.domain.value_objects import FetchRequest
from src.app.ohlcv.domain.entities import OHLCVCandle
from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe
from src.tests.conftest import END_DT, make_asset, make_candle, make_date_range, START_DT
from src.tests.ingestion.conftest import FakeMarketDataFetcher, FakeOHLCVRepository


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BTC: Asset = make_asset("BTCUSDT")
_ETH: Asset = make_asset("ETHUSDT")


# ---------------------------------------------------------------------------
# ingest_asset tests
# ---------------------------------------------------------------------------


class TestIngestionServiceIngestAsset:
    """Tests for ``IngestionService.ingest_asset()``."""

    def test_fetcher_returns_candles_repository_is_called(self) -> None:
        """When fetcher returns candles, repository.ingest() must be called with them."""
        candles: list[OHLCVCandle] = [
            make_candle(_BTC, Timeframe.H1, START_DT),
            make_candle(_BTC, Timeframe.H1, START_DT + timedelta(hours=1)),
        ]
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=candles)
        repo: FakeOHLCVRepository = FakeOHLCVRepository()

        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)
        cmd: IngestAssetCommand = IngestAssetCommand(
            asset=_BTC,
            timeframe=Timeframe.H1,
            date_range=make_date_range(),
        )

        result: int = service.ingest_asset(cmd)

        assert result == 2
        assert repo.ingest_call_count == 1
        assert len(repo.ingested) == 2

    def test_fetcher_returns_empty_list_repository_not_called(self) -> None:
        """When fetcher returns empty list, repository.ingest() must NOT be called."""
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[])
        repo: FakeOHLCVRepository = FakeOHLCVRepository()

        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)
        cmd: IngestAssetCommand = IngestAssetCommand(
            asset=_BTC,
            timeframe=Timeframe.H1,
            date_range=make_date_range(),
        )

        result: int = service.ingest_asset(cmd)

        assert result == 0
        assert repo.ingest_call_count == 0

    def test_fetcher_receives_correct_request(self) -> None:
        """ingest_asset() must construct a FetchRequest matching the command fields."""
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[])
        repo: FakeOHLCVRepository = FakeOHLCVRepository()
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        dr: DateRange = make_date_range()
        cmd: IngestAssetCommand = IngestAssetCommand(
            asset=_BTC,
            timeframe=Timeframe.H4,
            date_range=dr,
        )

        service.ingest_asset(cmd)

        assert len(fetcher.calls) == 1
        received_request: FetchRequest = fetcher.calls[0]
        assert received_request.asset == _BTC
        assert received_request.timeframe == Timeframe.H4
        assert received_request.date_range == dr

    def test_single_candle_is_stored(self) -> None:
        """ingest_asset() must correctly handle a single-candle response."""
        single_candle: list[OHLCVCandle] = [make_candle(_BTC, Timeframe.D1, START_DT)]
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=single_candle)
        repo: FakeOHLCVRepository = FakeOHLCVRepository()
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        cmd: IngestAssetCommand = IngestAssetCommand(
            asset=_BTC,
            timeframe=Timeframe.D1,
            date_range=make_date_range(),
        )
        result: int = service.ingest_asset(cmd)

        assert result == 1
        assert repo.ingested[0].asset.symbol == "BTCUSDT"


# ---------------------------------------------------------------------------
# ingest_universe tests
# ---------------------------------------------------------------------------


class TestIngestionServiceIngestUniverse:
    """Tests for ``IngestionService.ingest_universe()``."""

    def test_two_assets_two_timeframes_calls_ingest_four_times(self) -> None:
        """2 assets x 2 timeframes must trigger exactly 4 ingest_asset() calls."""
        candle: OHLCVCandle = make_candle(_BTC, Timeframe.H1, START_DT)
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[candle])
        repo: FakeOHLCVRepository = FakeOHLCVRepository()
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=[_BTC, _ETH],
            timeframes=[Timeframe.H1, Timeframe.H4],
            date_range=make_date_range(),
        )

        results: dict[str, int] = service.ingest_universe(cmd)

        assert len(fetcher.calls) == 4
        assert len(results) == 4

    def test_result_keys_use_symbol_slash_timeframe_format(self) -> None:
        """Result dict keys must follow 'SYMBOL/timeframe_value' format."""
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[])
        repo: FakeOHLCVRepository = FakeOHLCVRepository()
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=[_BTC],
            timeframes=[Timeframe.H1],
            date_range=make_date_range(),
        )

        results: dict[str, int] = service.ingest_universe(cmd)

        assert "BTCUSDT/1h" in results

    def test_result_counts_match_rows_returned(self) -> None:
        """Each result key's count must equal the number of candles fetched."""
        candles_per_call: list[OHLCVCandle] = [
            make_candle(_BTC, Timeframe.H1, START_DT + timedelta(hours=i)) for i in range(3)
        ]
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(
            candles_to_return=candles_per_call
        )
        repo: FakeOHLCVRepository = FakeOHLCVRepository()
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=[_BTC],
            timeframes=[Timeframe.H1],
            date_range=make_date_range(),
        )

        results: dict[str, int] = service.ingest_universe(cmd)

        assert results["BTCUSDT/1h"] == 3

    def test_single_asset_single_timeframe_returns_one_entry(self) -> None:
        """Universe with 1 asset and 1 timeframe must produce a dict with 1 entry."""
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[])
        repo: FakeOHLCVRepository = FakeOHLCVRepository()
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=[_BTC],
            timeframes=[Timeframe.D1],
            date_range=make_date_range(),
        )

        results: dict[str, int] = service.ingest_universe(cmd)

        assert len(results) == 1
        assert "BTCUSDT/1d" in results

    def test_all_result_values_are_zero_when_fetcher_returns_nothing(self) -> None:
        """When fetcher always returns empty, all result values must be 0."""
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[])
        repo: FakeOHLCVRepository = FakeOHLCVRepository()
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=[_BTC, _ETH],
            timeframes=[Timeframe.H1, Timeframe.H4, Timeframe.D1],
            date_range=make_date_range(),
        )

        results: dict[str, int] = service.ingest_universe(cmd)

        for count in results.values():
            assert count == 0


# ---------------------------------------------------------------------------
# ingest_incremental tests
# ---------------------------------------------------------------------------


class TestIngestionServiceIngestIncremental:
    """Tests for ``IngestionService.ingest_incremental()``."""

    def test_no_existing_data_fetches_full_range(self) -> None:
        """With no prior data, the full date_range must be fetched as-is."""
        candle: OHLCVCandle = make_candle(_BTC, Timeframe.H1, START_DT)
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[candle])
        repo: FakeOHLCVRepository = FakeOHLCVRepository(existing_date_range=None)
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        dr: DateRange = make_date_range()
        result: int = service.ingest_incremental(_BTC, Timeframe.H1, dr)

        assert result == 1
        assert len(fetcher.calls) == 1
        # The request start must equal the original date_range start
        sent_request: FetchRequest = fetcher.calls[0]
        assert sent_request.date_range.start == START_DT

    def test_existing_data_advances_start_date(self) -> None:
        """When prior data exists, the incremental start must be advanced past it."""
        existing_end: datetime = datetime(2024, 3, 1, tzinfo=UTC)
        existing_range: DateRange = DateRange(
            start=START_DT,
            end=existing_end,
        )
        candle: OHLCVCandle = make_candle(_BTC, Timeframe.H1, existing_end)
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[candle])
        repo: FakeOHLCVRepository = FakeOHLCVRepository(existing_date_range=existing_range)
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        full_range: DateRange = make_date_range(start=START_DT, end=END_DT)
        result: int = service.ingest_incremental(_BTC, Timeframe.H1, full_range)

        assert result == 1
        sent_request: FetchRequest = fetcher.calls[0]
        # Incremental start must be after the original START_DT
        assert sent_request.date_range.start > START_DT

    def test_already_up_to_date_returns_zero_without_fetching(self) -> None:
        """When existing data covers the full requested range, return 0 and skip fetching.

        The service computes ``incremental_start = existing_end - 1s`` and returns 0
        when ``incremental_start >= date_range.end``.  So the existing_end must be at
        least ``date_range.end + 1 second`` to satisfy the early-exit condition.
        """
        requested_end: datetime = datetime(2024, 3, 1, tzinfo=UTC)
        # existing_end must be > requested_end so that (existing_end - 1s) >= requested_end
        existing_end: datetime = requested_end + timedelta(seconds=2)
        existing_range: DateRange = DateRange(
            start=START_DT,
            end=existing_end,
        )
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[])
        repo: FakeOHLCVRepository = FakeOHLCVRepository(existing_date_range=existing_range)
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        full_range: DateRange = make_date_range(start=START_DT, end=requested_end)
        result: int = service.ingest_incremental(_BTC, Timeframe.H1, full_range)

        assert result == 0
        assert len(fetcher.calls) == 0

    def test_incremental_start_is_one_second_before_existing_end(self) -> None:
        """The incremental start must equal existing_end - 1 second (overlap protection)."""
        existing_end: datetime = datetime(2024, 3, 15, 12, 0, 0, tzinfo=UTC)
        expected_incremental_start: datetime = existing_end - timedelta(seconds=1)

        # Make the requested end far enough in the future so we do fetch
        requested_end: datetime = datetime(2024, 6, 1, tzinfo=UTC)
        existing_range: DateRange = DateRange(
            start=START_DT,
            end=existing_end,
        )
        candle: OHLCVCandle = make_candle(_BTC, Timeframe.H1, existing_end)
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[candle])
        repo: FakeOHLCVRepository = FakeOHLCVRepository(existing_date_range=existing_range)
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        full_range: DateRange = DateRange(start=START_DT, end=requested_end)
        service.ingest_incremental(_BTC, Timeframe.H1, full_range)

        sent_request: FetchRequest = fetcher.calls[0]
        assert sent_request.date_range.start == expected_incremental_start

    def test_existing_end_earlier_than_range_start_does_not_advance(self) -> None:
        """If incremental_start <= date_range.start, the original start must be kept."""
        # existing_end - 1s is still <= date_range.start, so no advancement
        early_end: datetime = datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC)  # just 1s past start
        existing_range: DateRange = DateRange(
            start=datetime(2023, 12, 1, tzinfo=UTC),
            end=early_end,
        )
        candle: OHLCVCandle = make_candle(_BTC, Timeframe.H1, START_DT)
        fetcher: FakeMarketDataFetcher = FakeMarketDataFetcher(candles_to_return=[candle])
        repo: FakeOHLCVRepository = FakeOHLCVRepository(existing_date_range=existing_range)
        service: IngestionService = IngestionService(fetcher=fetcher, repository=repo)

        # date_range start is START_DT = 2024-01-01 00:00:00
        # incremental_start = early_end - 1s = 2024-01-01 00:00:00 which is NOT > START_DT
        full_range: DateRange = make_date_range(start=START_DT, end=END_DT)
        service.ingest_incremental(_BTC, Timeframe.H1, full_range)

        sent_request: FetchRequest = fetcher.calls[0]
        # original start must be preserved (incremental_start not > date_range.start)
        assert sent_request.date_range.start == START_DT
