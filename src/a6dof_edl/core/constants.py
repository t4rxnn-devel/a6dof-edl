"""Physical and planetary constants (Earth, SI units).

All values follow the US Standard Atmosphere 1976 (NOAA-S/T 76-1562) and
the WGS-84 geopotential model, per Sections 2-3 of the specification.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Geopotential / planetary constants (Earth)
# ---------------------------------------------------------------------------

#: Standard gravitational parameter [m^3 s^-2]
MU_EARTH: float = 3.986004418e14

#: Mean equatorial radius [m]
R_EARTH: float = 6_378_137.0

#: Second zonal harmonic coefficient (J2, dimensionless)
J2: float = 1.08262668e-3

#: Earth rotation rate [rad s^-1]
OMEGA_EARTH: float = 7.2921150e-5

# ---------------------------------------------------------------------------
# US Standard Atmosphere 1976 constants
# ---------------------------------------------------------------------------

#: Universal gas constant [J mol^-1 K^-1]
R_STAR: float = 8314.32

#: Mean molecular weight of air [kg kmol^-1]
M0: float = 28.9644

#: Specific gas constant of air [J kg^-1 K^-1]  (R* / M0)
R_AIR: float = R_STAR / M0

#: Ratio of specific heats of air (dimensionless)
GAMMA_AIR: float = 1.40

#: Sea-level gravitational acceleration used by US76 [m s^-2]
G0: float = 9.80665

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

DEG2RAD: float = math.pi / 180.0
RAD2DEG: float = 180.0 / math.pi


def geopotential_altitude(h_geometric: float, radius: float = R_EARTH) -> float:
    """Convert geometric altitude to geopotential altitude.

    h_g = R_E * h / (R_E + h)   (Section 3, Eq. preamble)
    """
    return radius * h_geometric / (radius + h_geometric)


def geometric_altitude(h_geopotential: float, radius: float = R_EARTH) -> float:
    """Inverse of :func:`geopotential_altitude`."""
    return radius * h_geopotential / (radius - h_geopotential)
