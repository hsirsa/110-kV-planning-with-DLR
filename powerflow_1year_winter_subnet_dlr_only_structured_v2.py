import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlr_vc_v2.seasons import run_single_season_study


if __name__ == "__main__":
    run_single_season_study("winter")
