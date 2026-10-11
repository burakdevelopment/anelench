from __future__ import annotations

import math

import numpy as np

from .constants import C_LIGHT, ecef_to_geodetic


def gmst_deg(mjd: int, ut_hours: np.ndarray) -> np.ndarray:
    jd = mjd + 2400000.5 + np.asarray(ut_hours) / 24.0
    t = (jd - 2451545.0) / 36525.0
    g = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t**2 - t**3 / 38710000.0
    return np.mod(g, 360.0)


def elevation_deg(xyz: np.ndarray, ra_deg: float, dec_deg: float, mjd: int, ut_hours: np.ndarray) -> np.ndarray:
    lat, lon, _ = ecef_to_geodetic(np.asarray(xyz))
    h = np.radians(gmst_deg(mjd, ut_hours) + lon - ra_deg)
    phi, dec = math.radians(lat), math.radians(dec_deg)
    s = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * np.cos(h)
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def baseline_uv(xyz1: np.ndarray, xyz2: np.ndarray, ra_deg: float, dec_deg: float, mjd: int,
                ut_hours: np.ndarray, freq_hz: float) -> tuple[np.ndarray, np.ndarray]:
    b = (np.asarray(xyz1) - np.asarray(xyz2)) * freq_hz / C_LIGHT
    h = np.radians(gmst_deg(mjd, ut_hours) - ra_deg)
    dec = math.radians(dec_deg)
    u = np.sin(h) * b[0] + np.cos(h) * b[1]
    v = -math.sin(dec) * np.cos(h) * b[0] + math.sin(dec) * np.sin(h) * b[1] + math.cos(dec) * b[2]
    return u, v


def subsource_point(ra_deg: float, dec_deg: float, mjd: int, ut_hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.mod(ra_deg - gmst_deg(mjd, ut_hours) + 180.0, 360.0) - 180.0
    return np.full_like(lon, dec_deg), lon
