"""Shared benchmark categories and JSSP upper bounds for reporting scripts."""

from __future__ import annotations

CATEGORIES = ["DMU", "TA", "ABZ", "SWV", "YN", "P4", "P10"]

UPPER_BOUNDS = {
    "rcmax_20_15_5": 2731,
    "rcmax_20_15_8": 2669,
    "rcmax_20_20_7": 3188,
    "rcmax_20_20_8": 3092,
    "rcmax_30_15_5": 3681,
    "rcmax_30_15_4": 3394,
    "rcmax_30_20_9": 3844,
    "rcmax_30_20_8": 3764,
    "rcmax_40_15_10": 4668,
    "rcmax_40_15_8": 4648,
    "rcmax_40_20_6": 4692,
    "rcmax_40_20_2": 4691,
    "rcmax_50_15_2": 5385,
    "rcmax_50_15_4": 5385,
    "rcmax_50_20_6": 5713,
    "rcmax_50_20_9": 5747,
    "TA01": 1231,
    "TA02": 1039,
    "TA51": 1234,
    "TA52": 1000,
    "TA61": 1231,
    "TA71": 1659,
    "TA72": 1247,
    "abz07": 656,
    "abz08": 665,
    "abz09": 679,
    "swv01": 1397,
    "swv02": 1250,
    "swv03": 1268,
    "swv04": 616,
    "swv05": 1294,
    "swv06": 1268,
    "swv07": 1478,
    "swv08": 1500,
    "swv09": 1659,
    "swv10": 1234,
    "swv11": 1435,
    "swv12": 1794,
    "swv13": 1547,
    "swv14": 1000,
    "swv15": 1000,
    "yn01": 1165,
    "yn02": 1000,
    "yn03": 892,
    "yn04": 1165,
}


def categorize_dataset(dataset_name: str) -> str | None:
    if dataset_name.startswith("rcmax_"):
        return "DMU"
    if dataset_name.startswith("TA"):
        return "TA"
    if dataset_name.startswith("abz"):
        return "ABZ"
    if dataset_name.startswith("swv"):
        return "SWV"
    if dataset_name.startswith("yn"):
        return "YN"
    if dataset_name.startswith("p4_instance"):
        return "P4"
    if dataset_name.startswith("p10_instance"):
        return "P10"
    return None


def get_upper_bound(dataset_name: str) -> float:
    return float(UPPER_BOUNDS.get(dataset_name, 1000))


def makespan_gap_percent(makespan: float | None, dataset_name: str) -> float | None:
    if makespan is None or makespan <= 0:
        return None
    ub = get_upper_bound(dataset_name)
    if ub <= 0:
        return None
    return (makespan / ub) * 100.0
