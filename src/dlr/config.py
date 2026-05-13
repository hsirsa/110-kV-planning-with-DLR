from collections import OrderedDict
from pathlib import Path

GRID_CODE = "1-HV-mixed--0-no_sw"
TIME_STEP_HOURS = 0.25

# Resolved at import time so CSV paths work regardless of working directory.
_PROJECT_ROOT = Path(__file__).parent.parent

SUBNET_BUS_NAMES = [
    "EHV Bus 143",
    "HV1 Bus 5",
    "HV1 Bus 47",
    "HV1 Bus 21",
    "HV1 Bus 49",
    "HV1 Bus 67",
    "HV1 Bus 19",
    "HV1 Bus 35",
]
SUBNET_DIR_NAME = "subnet_focus"
WEATHER_DIR = Path("weather_tampere_dlr")
AITOLAHTI_WEATHER_CSV = WEATHER_DIR / "Aitolahti_Tampere_Finland_2023_dlr_weather_15min.csv"

CONDUCTOR_TEMP_C = 75.0
DEFAULT_EMISSIVITY = 0.8
DEFAULT_ABSORPTIVITY = 0.8
DEFAULT_ALPHA = 0.00403
SIGMA = 5.670374419e-8

ELECTRIC_BOILER_CONFIGS = [
    {
        "profile_csv": _PROJECT_ROOT / "electric_boiler_50MW_newprice.csv",
        "bus_name": "HV1 Bus 67",
        "load_name": "Electric boiler 50 MW",
    },
    {
        "profile_csv": _PROJECT_ROOT / "electric_boiler_100MW_newprice.csv",
        "bus_name": "HV1 Bus 35",
        "load_name": "Electric boiler 100 MW",
    },
]

SEASON_MONTHS = OrderedDict(
    [
        ("winter", [1, 2, 12]),
        ("spring", [3, 4, 5]),
        ("summer", [6, 7, 8]),
        ("autumn", [9, 10, 11]),
    ]
)
SEASONAL_OUTPUT_ROOT = "results_1year_seasons"
TOP_LINE_COUNT = 20
