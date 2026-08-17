"""US Standard Atmosphere 1976 — piecewise-continuous formulation.

Implements Section 3 of the specification, Eqs. (6)-(9), using the exact
layer architecture of Table 1:

  Layer  Base h_b [km]  Lapse L_b [K/km]  T_b [K]    P_b [Pa]
    0        0.00          -6.5           288.15   101325.0
    1       11.00           0.0           216.65    22632.0
    2       20.00          +1.0           216.65     5474.9
    3       32.00          +2.8           228.65      868.02
    4       47.00           0.0           270.65      110.91
    5       51.00          -2.8           270.65       66.938
    6       71.00          -2.0           214.65        3.9564
    7       84.85           0.0           186.87        0.3734

Geopotential altitude h_g = R_E h / (R_E + h) is used for the layer
evaluation, per Section 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from a6dof_edl.core.constants import (
    GAMMA_AIR,
    G0,
    M0,
    R_AIR,
    R_STAR,
    geopotential_altitude,
)

# Layer base geopotential altitudes [m], lapse rates [K/m],
# base temperatures [K], base pressures [Pa] — Table 1.
_H_B = np.array([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.85]) * 1e3
_L_B = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0]) * 1e-3
_T_B = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87])
_P_B = np.array([101325.0, 22632.0, 5474.9, 868.02, 110.91, 66.938, 3.9564, 0.3734])

_H_TOP = 84.85e3  # top of tabulated regime [m geopotential]


@dataclass(frozen=True)
class AtmosphereState:
    """Atmospheric state at a single altitude."""

    altitude_geometric_m: float
    altitude_geopotential_m: float
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float


class USStandardAtmosphere1976:
    """Piecewise-continuous US76 model (0 - 84.85 km geopotential).

    Above the tabulated regime an exponential extrapolation on the layer-7
    isothermal temperature is applied; below sea level the layer-0 gradient
    extrapolation is clamped to avoid nonphysical temperatures.
    """

    def __init__(self, density_scale: float = 1.0) -> None:
        # density_scale: multiplicative factor for Monte Carlo dispersions.
        if density_scale <= 0.0:
            raise ValueError("density_scale must be positive.")
        self.density_scale = float(density_scale)

    # ------------------------------------------------------------------
    def _layer_index(self, h_g: float) -> int:
        return int(np.clip(np.searchsorted(_H_B, h_g, side="right") - 1, 0, 7))

    # ------------------------------------------------------------------
    def properties(self, h_geometric: float) -> AtmosphereState:
        """Full atmospheric state at geometric altitude h_geometric [m]."""
        h_g = geopotential_altitude(max(h_geometric, -500.0))

        if h_g <= _H_TOP:
            i = self._layer_index(max(h_g, 0.0))
            T = _T_B[i] + _L_B[i] * (h_g - _H_B[i])  # Eq. (6)
            if _L_B[i] != 0.0:
                # Eq. (7): gradient layers
                P = _P_B[i] * (_T_B[i] / T) ** (G0 * M0 / (R_STAR * _L_B[i]))
            else:
                # Eq. (8): isothermal layers
                P = _P_B[i] * np.exp(-G0 * M0 * (h_g - _H_B[i]) / (R_STAR * _T_B[i]))
        else:
            # Exponential continuation on the layer-7 isothermal slab.
            T = _T_B[7]
            P = _P_B[7] * np.exp(-G0 * M0 * (h_g - _H_B[7]) / (R_STAR * _T_B[7]))

        T = max(T, 150.0)  # numerical floor, far below any encountered value
        rho = self.density_scale * P / (R_AIR * T)  # Eq. (9), left
        a = np.sqrt(GAMMA_AIR * R_AIR * T)          # Eq. (9), right
        return AtmosphereState(
            altitude_geometric_m=h_geometric,
            altitude_geopotential_m=h_g,
            temperature_K=float(T),
            pressure_Pa=float(P),
            density_kg_m3=float(rho),
            speed_of_sound_m_s=float(a),
        )

    # Convenience scalar accessors --------------------------------------
    def density(self, h_geometric: float) -> float:
        return self.properties(h_geometric).density_kg_m3

    def temperature(self, h_geometric: float) -> float:
        return self.properties(h_geometric).temperature_K

    def pressure(self, h_geometric: float) -> float:
        return self.properties(h_geometric).pressure_Pa

    def speed_of_sound(self, h_geometric: float) -> float:
        return self.properties(h_geometric).speed_of_sound_m_s
