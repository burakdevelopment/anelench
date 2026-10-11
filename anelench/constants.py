from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

C_LIGHT = 299_792_458.0
RAD_PER_UAS = math.pi / (180.0 * 3600.0 * 1e6)
UAS_PER_RAD = 1.0 / RAD_PER_UAS
ETA_QUANT = 0.88

M87_RA_DEG = 187.7059307575226
M87_DEC_DEG = 12.39112323919932

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    return np.array([
        (n + alt_m) * math.cos(lat) * math.cos(lon),
        (n + alt_m) * math.cos(lat) * math.sin(lon),
        (n * (1.0 - _WGS84_E2) + alt_m) * math.sin(lat),
    ])


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (float(c) for c in xyz)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(6):
        n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
        alt = p / max(math.cos(lat), 1e-12) - n
        lat = math.atan2(z, p * (1.0 - _WGS84_E2 * n / (n + alt)))
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    alt = p / max(math.cos(lat), 1e-12) - n
    return math.degrees(lat), math.degrees(lon), alt


@dataclass(frozen=True)
class Station:
    code: str
    name: str
    location: str
    xyz: tuple[float, float, float]
    sefd: float
    cost_weight: float
    in_2017: bool
    joined: int

    @property
    def lat_lon(self) -> tuple[float, float]:
        lat, lon, _ = ecef_to_geodetic(np.asarray(self.xyz))
        return lat, lon


STATIONS: dict[str, Station] = {
    s.code: s
    for s in [
        Station("AA", "ALMA", "Chile", (2225060.8136, -5440059.59994, -2481681.15054), 90.0, 3.0, True, 2017),
        Station("AP", "APEX", "Chile", (2225039.5297, -5441197.6292, -2479303.3597), 3500.0, 1.0, True, 2017),
        Station("AZ", "SMT", "Arizona, USA", (-1828796.2, -5054406.8, 3427865.2), 11000.0, 1.0, True, 2017),
        Station("JC", "JCMT", "Hawai'i, USA", (-5464584.676, -2493001.17, 2150653.982), 6000.0, 1.0, True, 2017),
        Station("LM", "LMT", "Mexico", (-768715.632, -5988507.072, 2063354.852), 2000.0, 1.0, True, 2017),
        Station("PV", "IRAM 30m", "Spain", (5088967.74544, -301681.18586, 3825012.20561), 1500.0, 1.0, True, 2017),
        Station("SM", "SMA", "Hawai'i, USA", (-5464555.493, -2492927.989, 2150797.176), 4500.0, 1.5, True, 2017),
        Station("GL", "Greenland Telescope", "Greenland", tuple(geodetic_to_ecef(76.5312, -68.6858, 102.0)), 7000.0, 1.0, False, 2018),
        Station("KP", "Kitt Peak 12m", "Arizona, USA", tuple(geodetic_to_ecef(31.9533, -111.6150, 1914.0)), 11000.0, 1.0, False, 2021),
        Station("NN", "NOEMA", "France", tuple(geodetic_to_ecef(44.6339, 5.9079, 2550.0)), 700.0, 2.0, False, 2021),
    ]
}

INTRA_SITE = {frozenset({"AA", "AP"}), frozenset({"JC", "SM"}), frozenset({"SM", "SR"}), frozenset({"JC", "SR"})}
