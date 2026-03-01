"""Unit tests for ingestion application command objects.

Tests cover ``IngestAssetCommand`` and ``IngestUniverseCommand``
including immutability and validation constraints.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.app.ingestion.application.commands import IngestAssetCommand, IngestUniverseCommand
from src.app.ohlcv.domain.value_objects import Asset, Timeframe
from src.tests.conftest import END_DT, START_DT, make_asset, make_date_range


# ---------------------------------------------------------------------------
# IngestAssetCommand
# ---------------------------------------------------------------------------


class TestIngestAssetCommand:
    """Tests for the ``IngestAssetCommand`` Pydantic model."""

    def test_construction_with_valid_fields(self) -> None:
        """IngestAssetCommand must be constructable from valid asset, timeframe, date_range."""
        asset: Asset = make_asset("BTCUSDT")
        timeframe: Timeframe = Timeframe.H1
        dr = make_date_range()

        cmd: IngestAssetCommand = IngestAssetCommand(
            asset=asset,
            timeframe=timeframe,
            date_range=dr,
        )

        assert cmd.asset.symbol == "BTCUSDT"
        assert cmd.timeframe == Timeframe.H1
        assert cmd.date_range.start == START_DT
        assert cmd.date_range.end == END_DT

    def test_frozen_field_assignment_raises_validation_error(self) -> None:
        """Mutating a field on a frozen IngestAssetCommand must raise ValidationError."""
        cmd: IngestAssetCommand = IngestAssetCommand(
            asset=make_asset("BTCUSDT"),
            timeframe=Timeframe.H1,
            date_range=make_date_range(),
        )

        with pytest.raises(ValidationError):
            cmd.timeframe = Timeframe.D1  # type: ignore[misc]

    def test_frozen_asset_assignment_raises_validation_error(self) -> None:
        """Mutating the asset field on a frozen command must raise ValidationError."""
        cmd: IngestAssetCommand = IngestAssetCommand(
            asset=make_asset("BTCUSDT"),
            timeframe=Timeframe.H1,
            date_range=make_date_range(),
        )

        with pytest.raises(ValidationError):
            cmd.asset = make_asset("ETHUSDT")  # type: ignore[misc]

    def test_missing_asset_raises_validation_error(self) -> None:
        """Omitting the asset field must raise ValidationError."""
        with pytest.raises(ValidationError):
            IngestAssetCommand(  # type: ignore[call-arg]
                timeframe=Timeframe.H1,
                date_range=make_date_range(),
            )

    def test_missing_timeframe_raises_validation_error(self) -> None:
        """Omitting the timeframe field must raise ValidationError."""
        with pytest.raises(ValidationError):
            IngestAssetCommand(  # type: ignore[call-arg]
                asset=make_asset(),
                date_range=make_date_range(),
            )

    def test_missing_date_range_raises_validation_error(self) -> None:
        """Omitting the date_range field must raise ValidationError."""
        with pytest.raises(ValidationError):
            IngestAssetCommand(  # type: ignore[call-arg]
                asset=make_asset(),
                timeframe=Timeframe.H1,
            )

    @pytest.mark.parametrize("timeframe", list(Timeframe))
    def test_construction_with_all_timeframes(self, timeframe: Timeframe) -> None:
        """IngestAssetCommand must accept every supported Timeframe variant."""
        cmd: IngestAssetCommand = IngestAssetCommand(
            asset=make_asset(),
            timeframe=timeframe,
            date_range=make_date_range(),
        )
        assert cmd.timeframe == timeframe

    def test_hash_equality_for_identical_commands(self) -> None:
        """Two identical frozen commands must be equal (frozen Pydantic models are hashable)."""
        cmd_a: IngestAssetCommand = IngestAssetCommand(
            asset=make_asset("BTCUSDT"),
            timeframe=Timeframe.H1,
            date_range=make_date_range(),
        )
        cmd_b: IngestAssetCommand = IngestAssetCommand(
            asset=make_asset("BTCUSDT"),
            timeframe=Timeframe.H1,
            date_range=make_date_range(),
        )
        assert cmd_a == cmd_b


# ---------------------------------------------------------------------------
# IngestUniverseCommand
# ---------------------------------------------------------------------------


class TestIngestUniverseCommand:
    """Tests for the ``IngestUniverseCommand`` Pydantic model."""

    def test_construction_with_valid_fields(self) -> None:
        """IngestUniverseCommand must be constructable with valid assets, timeframes, date_range."""
        assets: list[Asset] = [make_asset("BTCUSDT"), make_asset("ETHUSDT")]
        timeframes: list[Timeframe] = [Timeframe.H1, Timeframe.H4]
        dr = make_date_range()

        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=assets,
            timeframes=timeframes,
            date_range=dr,
        )

        assert len(cmd.assets) == 2
        assert len(cmd.timeframes) == 2
        assert cmd.assets[0].symbol == "BTCUSDT"
        assert cmd.assets[1].symbol == "ETHUSDT"
        assert Timeframe.H1 in cmd.timeframes
        assert Timeframe.H4 in cmd.timeframes

    def test_empty_assets_list_raises_validation_error(self) -> None:
        """Providing an empty assets list must raise ValidationError (min_length=1)."""
        with pytest.raises(ValidationError, match="at least 1 item"):
            IngestUniverseCommand(
                assets=[],
                timeframes=[Timeframe.H1],
                date_range=make_date_range(),
            )

    def test_empty_timeframes_list_raises_validation_error(self) -> None:
        """Providing an empty timeframes list must raise ValidationError (min_length=1)."""
        with pytest.raises(ValidationError, match="at least 1 item"):
            IngestUniverseCommand(
                assets=[make_asset()],
                timeframes=[],
                date_range=make_date_range(),
            )

    def test_frozen_assets_assignment_raises_validation_error(self) -> None:
        """Mutating the assets field on a frozen command must raise ValidationError."""
        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=[make_asset("BTCUSDT")],
            timeframes=[Timeframe.H1],
            date_range=make_date_range(),
        )

        with pytest.raises(ValidationError):
            cmd.assets = [make_asset("ETHUSDT")]  # type: ignore[misc]

    def test_frozen_timeframes_assignment_raises_validation_error(self) -> None:
        """Mutating the timeframes field on a frozen command must raise ValidationError."""
        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=[make_asset()],
            timeframes=[Timeframe.H1],
            date_range=make_date_range(),
        )

        with pytest.raises(ValidationError):
            cmd.timeframes = [Timeframe.D1]  # type: ignore[misc]

    def test_single_asset_single_timeframe_is_valid(self) -> None:
        """The minimum valid universe (1 asset, 1 timeframe) must be accepted."""
        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=[make_asset("BTCUSDT")],
            timeframes=[Timeframe.D1],
            date_range=make_date_range(),
        )
        assert len(cmd.assets) == 1
        assert len(cmd.timeframes) == 1

    def test_all_timeframes_accepted(self) -> None:
        """IngestUniverseCommand must accept a list containing every Timeframe variant."""
        all_timeframes: list[Timeframe] = list(Timeframe)
        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=[make_asset()],
            timeframes=all_timeframes,
            date_range=make_date_range(),
        )
        assert len(cmd.timeframes) == len(all_timeframes)

    def test_large_universe_is_accepted(self) -> None:
        """IngestUniverseCommand must accept many assets and timeframes without error."""
        many_assets: list[Asset] = [
            Asset(symbol=f"ASSET{i:02d}") for i in range(10)
        ]
        all_timeframes: list[Timeframe] = list(Timeframe)

        cmd: IngestUniverseCommand = IngestUniverseCommand(
            assets=many_assets,
            timeframes=all_timeframes,
            date_range=make_date_range(),
        )

        assert len(cmd.assets) == 10
        assert len(cmd.timeframes) == len(all_timeframes)
