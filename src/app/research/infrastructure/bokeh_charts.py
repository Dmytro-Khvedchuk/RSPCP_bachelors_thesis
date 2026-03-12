"""Interactive Bokeh chart functions for research analysis visualisation.

Provides standalone chart-building functions that return Bokeh ``Figure`` or
``DataTable`` objects.  Charts cover data-quality heatmaps, gap timelines,
volume profiles, bar-count histograms, and sortable statistics tables.
"""

from __future__ import annotations

from math import nan
from typing import Final

import pandas as pd
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    DataTable,
    HoverTool,
    LinearColorMapper,
    NumberFormatter,
    StringFormatter,
    TableColumn,
)
from bokeh.palettes import Spectral11
from bokeh.plotting import figure
from bokeh.plotting._figure import figure as Figure  # noqa: N812

from src.app.research.domain.value_objects import GapRecord, ReturnStatistics

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

_PALETTE: Final[list[str]] = list(Spectral11)
_DEFAULT_WIDTH: Final[int] = 900
_DEFAULT_HEIGHT: Final[int] = 500
_ROLLING_WINDOW: Final[int] = 20


# ---------------------------------------------------------------------------
# Coverage heatmap
# ---------------------------------------------------------------------------


def create_coverage_heatmap(
    matrix: pd.DataFrame,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> Figure:
    """Create a rect-based heatmap of coverage percentages.

    The input is a pivot table where rows are assets, columns are timeframes,
    and values are coverage percentages (0-100).

    Args:
        matrix: Pivot DataFrame (index=assets, columns=timeframes,
            values=coverage_pct).
        width: Plot width in pixels. Defaults to 900.
        height: Plot height in pixels. Defaults to 500.

    Returns:
        Bokeh ``Figure`` with a colour-mapped heatmap and hover tooltips.
    """
    assets: list[str] = list(matrix.index.astype(str))
    timeframes: list[str] = list(matrix.columns.astype(str))

    # Flatten the pivot into parallel lists for ColumnDataSource.
    x_vals: list[str] = []
    y_vals: list[str] = []
    values: list[float] = []

    for asset in assets:
        for tf in timeframes:
            x_vals.append(tf)
            y_vals.append(asset)
            raw_value: float = float(matrix.loc[asset, tf])
            values.append(raw_value if pd.notna(raw_value) else nan)

    source: ColumnDataSource = ColumnDataSource(
        data={"timeframe": x_vals, "asset": y_vals, "coverage": values},
    )

    mapper: LinearColorMapper = LinearColorMapper(
        palette=_PALETTE,
        low=0.0,
        high=100.0,
        nan_color="grey",
    )

    fig: Figure = figure(
        title="Coverage Heatmap",
        x_range=timeframes,
        y_range=assets,
        width=width,
        height=height,
        toolbar_location="above",
        x_axis_label="Timeframe",
        y_axis_label="Asset",
    )

    fig.rect(
        x="timeframe",
        y="asset",
        width=1,
        height=1,
        source=source,
        fill_color={"field": "coverage", "transform": mapper},
        line_color=None,
    )

    color_bar: ColorBar = ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(desired_num_ticks=len(_PALETTE)),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
    )
    fig.add_layout(color_bar, "right")

    hover: HoverTool = HoverTool(
        tooltips=[
            ("Asset", "@asset"),
            ("Timeframe", "@timeframe"),
            ("Coverage %", "@coverage{0.1}"),
        ],
    )
    fig.add_tools(hover)

    return fig


# ---------------------------------------------------------------------------
# Gap timeline
# ---------------------------------------------------------------------------


def create_gap_timeline(
    gaps: list[GapRecord],
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> Figure:
    """Create a horizontal bar chart showing gap periods on a timeline.

    Each gap is rendered as an ``hbar`` glyph.  The Y-axis shows assets and
    the X-axis shows datetime.

    Args:
        gaps: List of ``GapRecord`` value objects to visualise.
        width: Plot width in pixels. Defaults to 900.
        height: Plot height in pixels. Defaults to 500.

    Returns:
        Bokeh ``Figure`` with horizontal bars and hover tooltips.
    """
    assets: list[str] = [g.asset for g in gaps]
    left_vals: list[float] = [g.gap_start.timestamp() * 1000.0 for g in gaps]
    right_vals: list[float] = [g.gap_end.timestamp() * 1000.0 for g in gaps]
    durations: list[float] = [g.gap_duration_hours for g in gaps]
    missing: list[int] = [g.missing_bars for g in gaps]
    timeframes: list[str] = [g.timeframe for g in gaps]

    source: ColumnDataSource = ColumnDataSource(
        data={
            "asset": assets,
            "left": left_vals,
            "right": right_vals,
            "duration_hours": durations,
            "missing_bars": missing,
            "timeframe": timeframes,
        },
    )

    unique_assets: list[str] = sorted(set(assets))

    fig: Figure = figure(
        title="Gap Timeline",
        y_range=unique_assets,
        x_axis_type="datetime",
        width=width,
        height=height,
        toolbar_location="above",
        x_axis_label="Time",
        y_axis_label="Asset",
    )

    fig.hbar(
        y="asset",
        left="left",
        right="right",
        height=0.4,
        source=source,
        color="#e74c3c",
        alpha=0.7,
    )

    hover: HoverTool = HoverTool(
        tooltips=[
            ("Asset", "@asset"),
            ("Timeframe", "@timeframe"),
            ("Duration (h)", "@duration_hours{0.1}"),
            ("Missing bars", "@missing_bars"),
        ],
    )
    fig.add_tools(hover)

    return fig


# ---------------------------------------------------------------------------
# Volume profile
# ---------------------------------------------------------------------------


def create_volume_profile(
    df: pd.DataFrame,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    rolling_window: int = _ROLLING_WINDOW,
) -> Figure:
    """Create a volume-over-time line chart with a rolling average overlay.

    Args:
        df: DataFrame with columns ``timestamp`` and ``volume``.
        width: Plot width in pixels. Defaults to 900.
        height: Plot height in pixels. Defaults to 500.
        rolling_window: Window size for the rolling mean. Defaults to 20.

    Returns:
        Bokeh ``Figure`` with volume line and rolling average, and
        pan/zoom enabled.
    """
    rolling_avg: pd.Series = df["volume"].rolling(window=rolling_window).mean()  # type: ignore[type-arg]

    source: ColumnDataSource = ColumnDataSource(
        data={
            "timestamp": df["timestamp"],
            "volume": df["volume"],
            "rolling_avg": rolling_avg,
        },
    )

    fig: Figure = figure(
        title="Volume Profile",
        x_axis_type="datetime",
        width=width,
        height=height,
        toolbar_location="above",
        x_axis_label="Time",
        y_axis_label="Volume",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )

    fig.line(
        x="timestamp",
        y="volume",
        source=source,
        color="#3498db",
        alpha=0.6,
        legend_label="Volume",
    )
    fig.line(
        x="timestamp",
        y="rolling_avg",
        source=source,
        color="#e74c3c",
        line_width=2,
        legend_label=f"Rolling Avg ({rolling_window})",
    )

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    hover: HoverTool = HoverTool(
        tooltips=[
            ("Time", "@timestamp{%F %T}"),
            ("Volume", "@volume{0.2f}"),
            ("Rolling Avg", "@rolling_avg{0.2f}"),
        ],
        formatters={"@timestamp": "datetime"},
    )
    fig.add_tools(hover)

    return fig


# ---------------------------------------------------------------------------
# Bar count histogram
# ---------------------------------------------------------------------------


def create_bar_count_histogram(
    weekly_counts: dict[str, pd.Series],  # type: ignore[type-arg]
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> Figure:
    """Create a multi-series step chart comparing weekly bar counts.

    Each key in ``weekly_counts`` is a bar type name; the value is a
    ``pd.Series`` indexed by week (datetime or period) with bar counts
    as values.

    Args:
        weekly_counts: Mapping from bar type name to weekly bar count Series.
        width: Plot width in pixels. Defaults to 900.
        height: Plot height in pixels. Defaults to 500.

    Returns:
        Bokeh ``Figure`` with one step line per bar type and a legend.
    """
    fig: Figure = figure(
        title="Weekly Bar Counts by Type",
        x_axis_type="datetime",
        width=width,
        height=height,
        toolbar_location="above",
        x_axis_label="Week",
        y_axis_label="Bar Count",
    )

    colors: list[str] = _PALETTE[: len(weekly_counts)]

    for idx, (bar_type, counts) in enumerate(weekly_counts.items()):
        color: str = colors[idx % len(colors)]

        # Convert index to datetime if it's a PeriodIndex.
        index_values: pd.Index = counts.index  # type: ignore[type-arg]
        if hasattr(index_values, "to_timestamp"):
            index_values = index_values.to_timestamp()  # type: ignore[union-attr]

        source: ColumnDataSource = ColumnDataSource(
            data={"week": index_values, "count": counts.values},
        )

        fig.step(
            x="week",
            y="count",
            source=source,
            color=color,
            line_width=2,
            legend_label=bar_type,
            mode="after",
        )

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    return fig


# ---------------------------------------------------------------------------
# Statistics table
# ---------------------------------------------------------------------------


def create_statistics_table(
    stats: list[ReturnStatistics],
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = 300,
) -> DataTable:
    """Create an interactive sortable Bokeh DataTable of return statistics.

    Args:
        stats: List of ``ReturnStatistics`` value objects.
        width: Table width in pixels. Defaults to 900.
        height: Table height in pixels. Defaults to 300.

    Returns:
        Bokeh ``DataTable`` with columns: asset, bar_type, count, mean,
        std, skewness, kurtosis, jarque_bera_stat, jarque_bera_pvalue,
        is_normal.
    """
    data: dict[str, list[object]] = {
        "asset": [s.asset for s in stats],
        "bar_type": [s.bar_type for s in stats],
        "count": [s.count for s in stats],
        "mean": [s.mean for s in stats],
        "std": [s.std for s in stats],
        "skewness": [s.skewness for s in stats],
        "kurtosis": [s.kurtosis for s in stats],
        "jarque_bera_stat": [s.jarque_bera_stat for s in stats],
        "jarque_bera_pvalue": [s.jarque_bera_pvalue for s in stats],
        "is_normal": [str(s.is_normal) for s in stats],
    }
    source: ColumnDataSource = ColumnDataSource(data=data)

    number_fmt: NumberFormatter = NumberFormatter(format="0.000000")
    string_fmt: StringFormatter = StringFormatter()

    columns: list[TableColumn] = [
        TableColumn(field="asset", title="Asset", formatter=string_fmt),
        TableColumn(field="bar_type", title="Bar Type", formatter=string_fmt),
        TableColumn(field="count", title="Count"),
        TableColumn(field="mean", title="Mean", formatter=number_fmt),
        TableColumn(field="std", title="Std", formatter=number_fmt),
        TableColumn(field="skewness", title="Skewness", formatter=number_fmt),
        TableColumn(field="kurtosis", title="Kurtosis", formatter=number_fmt),
        TableColumn(field="jarque_bera_stat", title="JB Stat", formatter=number_fmt),
        TableColumn(field="jarque_bera_pvalue", title="JB p-value", formatter=number_fmt),
        TableColumn(field="is_normal", title="Normal?", formatter=string_fmt),
    ]

    table: DataTable = DataTable(
        source=source,
        columns=columns,
        width=width,
        height=height,
        sortable=True,
    )

    return table
