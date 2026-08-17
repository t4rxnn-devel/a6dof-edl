"""Geopotential gravity model with J2 zonal harmonic (Section 2.2, Eq. 3).

g(r) = -mu/r^3 r + (3 mu J2 R_E^2 / 2 r^5) * [ x(5z^2/r^2 - 1),
                                               y(5z^2/r^2 - 1),
                                               z(5z^2/r^2 - 3) ]
"""

from __future__ import annotations

import numpy as np

from a6dof_edl.core.constants import J2, MU_EARTH, R_EARTH


class J2GravityModel:
    """Central-body + J2 gravitational acceleration in ECI coordinates."""

    def __init__(
        self,
        mu: float = MU_EARTH,
        j2: float = J2,
        radius: float = R_EARTH,
        enable_j2: bool = True,
    ) -> None:
        self.mu = float(mu)
        self.j2 = float(j2)
        self.radius = float(radius)
        self.enable_j2 = bool(enable_j2)

    # ------------------------------------------------------------------
    def acceleration(self, r_eci: np.ndarray) -> np.ndarray:
        """Gravitational acceleration [m/s^2] at ECI position r [m]."""
        x, y, z = r_eci
        r = np.linalg.norm(r_eci)
        if r < 1.0:
            raise ValueError("Position magnitude below 1 m; singular gravity evaluation.")

        a = -self.mu / r**3 * r_eci  # central body term

        if self.enable_j2:
            z2_r2 = (z / r) ** 2
            factor = 1.5 * self.mu * self.j2 * self.radius**2 / r**5
            a_j2 = factor * np.array([
                x * (5.0 * z2_r2 - 1.0),
                y * (5.0 * z2_r2 - 1.0),
                z * (5.0 * z2_r2 - 3.0),
            ])
            a = a + a_j2
        return a

    def magnitude(self, r_eci: np.ndarray) -> float:
        return float(np.linalg.norm(self.acceleration(r_eci)))
