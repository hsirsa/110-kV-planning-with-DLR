import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlr_vc_v2.seasons import run_single_season_study


def main():
    parser = argparse.ArgumentParser(description="Run one structured v2 seasonal subnet DLR study.")
    parser.add_argument("season", choices=["winter", "spring", "summer", "autumn"])
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    run_single_season_study(args.season, args.output_root)


if __name__ == "__main__":
    main()
