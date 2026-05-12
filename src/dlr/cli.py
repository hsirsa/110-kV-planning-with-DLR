import os
import warnings

import click

from .config import AITOLAHTI_WEATHER_CSV, SEASON_MONTHS, SEASONAL_OUTPUT_ROOT


@click.group()
def cli():
    """DLR network planning for the SimBench 110 kV HV1 grid."""


# ---------------------------------------------------------------------------
# Power flow commands
# ---------------------------------------------------------------------------

@cli.command("run-day")
@click.option("--output-root", default="results_1day", show_default=True, help="Output directory.")
def run_day_cmd(output_root):
    """Run a 1-day (96 time step) power flow study."""
    from .study import run_day_study
    run_day_study(output_root)


@cli.command("run-year")
@click.option("--output-root", default="results_1year", show_default=True, help="Output directory.")
def run_year_cmd(output_root):
    """Run a full 1-year power flow study (no seasons, no DLR)."""
    from .study import run_year_study
    run_year_study(output_root)


@cli.command("run")
@click.option(
    "--season",
    type=click.Choice(list(SEASON_MONTHS)),
    default=None,
    help="Run only one season. Omit to run all four.",
)
@click.option(
    "--output-root",
    default=SEASONAL_OUTPUT_ROOT,
    show_default=True,
    help="Root directory for seasonal results.",
)
def run_cmd(season, output_root):
    """Run seasonal power flow studies with DLR."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    import pandas as pd
    from .seasons import (
        build_season_time_steps,
        build_time_lookup,
        export_cross_season_summary_tables,
        load_weather_calendar,
        run_season_study,
        run_single_season_study,
    )

    if season:
        run_single_season_study(season, output_root)
        return

    max_steps = len(pd.read_csv(AITOLAHTI_WEATHER_CSV, usecols=[0]))
    weather_calendar = load_weather_calendar(max_steps)
    time_lookup = build_time_lookup(weather_calendar)
    season_time_steps = build_season_time_steps(weather_calendar)

    os.makedirs(output_root, exist_ok=True)
    for season_name, time_steps in season_time_steps.items():
        click.echo(f"Running {season_name} ({len(time_steps)} time steps)...")
        run_season_study(season_name, time_steps, output_root, time_lookup)
        click.echo(f"  → {os.path.join(output_root, season_name)}")

    export_cross_season_summary_tables(output_root)
    click.echo("All seasons complete.")


@cli.command("summary")
@click.option(
    "--output-root",
    default=SEASONAL_OUTPUT_ROOT,
    show_default=True,
    help="Root directory containing completed seasonal results.",
)
def summary_cmd(output_root):
    """Collect cross-season comparison tables from existing results."""
    from .seasons import export_cross_season_summary_tables
    export_cross_season_summary_tables(output_root)
    click.echo(f"Cross-season summary tables written to: {output_root}")


# ---------------------------------------------------------------------------
# Topology / diagram commands
# ---------------------------------------------------------------------------

@cli.command("topology")
@click.option("--output-dir", default="HV1_export", show_default=True, help="Output directory.")
def topology_cmd(output_dir):
    """Export network topology CSVs and a geo-coordinate diagram."""
    from .topology import run_geo_export
    run_geo_export(output_dir)


@cli.command("diagram")
@click.option("--output-dir", default="hv1_diagram_like_image", show_default=True, help="Output directory.")
def diagram_cmd(output_dir):
    """Export a BFS tree-layout network diagram (PNG + PDF)."""
    from .topology import run_tree_diagram
    run_tree_diagram(output_dir)


# ---------------------------------------------------------------------------
# Weather download command
# ---------------------------------------------------------------------------

@cli.command("weather")
@click.option("--year", default=2023, show_default=True, help="Year to download.")
@click.option(
    "--output-dir",
    default="weather_tampere_dlr",
    show_default=True,
    help="Directory for downloaded weather CSVs.",
)
def weather_cmd(year, output_dir):
    """Download ERA5 weather data (CDS primary, Open-Meteo fallback)."""
    from pathlib import Path
    from .weather import run_weather_download
    run_weather_download(year=year, out_dir=Path(output_dir))
