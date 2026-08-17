"""Vehicle aerodynamic model (Section 5.1, Eqs. 14-15).

C_L(M, alpha) = C_L0(alpha) + C_L,M(M)
C_D(M, alpha) = C_D0(alpha) + C_L^2 / (pi e AR)

Baseline trim: alpha_trim = 28 deg, hypersonic L/D ~= 0.35.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from a6dof_edl.core.constants import DEG2RAD


@dataclass(frozen=True)
class AeroState:
    """Resolved aerodynamic coefficients and derived quantities."""

    C_L: float
    C_D: float
    L_over_D: float
    dynamic_pressure_Pa: float
    mach: float


class EntryVehicleAerodynamics:
    """Mach/angle-of-attack dependent aerodynamic coefficient model.

    The hypersonic baseline uses Newtonian-inspired alpha dependence with a
    transonic C_L,M correction ramping between Mach 1.2 and 4.0, matched so
    the trim point (alpha = 28 deg) yields L/D ~= 0.35 at hypersonic Mach.
    """

    def __init__(
        self,
        reference_area_m2: float = 35.0,
        aspect_ratio: float = 1.4,
        oswald_efficiency: float = 0.75,
        alpha_trim_deg: float = 28.0,
        ld_trim_hypersonic: float = 0.35,
        ld_bias: float = 0.0,
    ) -> None:
        if reference_area_m2 <= 0.0:
            raise ValueError("reference_area_m2 must be positive.")
        self.S_ref = float(reference_area_m2)
        self.AR = float(aspect_ratio)
        self.e = float(oswald_efficiency)
        self.alpha_trim = alpha_trim_deg * DEG2RAD
        # ld_bias: additive L/D perturbation for Monte Carlo dispersions.
        self.ld_bias = float(ld_bias)

        # C_D0(alpha_trim) solved so that trim L/D hits the target at
        # hypersonic Mach, where C_L,M saturates at +0.10 (Eq. 14).
        s, c = np.sin(self.alpha_trim), np.cos(self.alpha_trim)
        C_L0_trim = 1.10 * s * s * c + 0.06 + 0.10  # full hypersonic trim C_L
        C_D_ind_trim = C_L0_trim**2 / (np.pi * self.e * self.AR)
        C_D_tot_trim = C_L0_trim / ld_trim_hypersonic
        self._C_D0_offset = C_D_tot_trim - C_D_ind_trim
        self._C_L0_trim = C_L0_trim

    # ------------------------------------------------------------------
    def _C_L0(self, alpha: float) -> float:
        return 1.10 * np.sin(alpha) ** 2 * np.cos(alpha) + 0.06

    def _C_L_M(self, mach: float) -> float:
        """Transonic lift correction C_L,M(M); zero beyond Mach 4."""
        if mach <= 1.2:
            return 0.0
        if mach >= 4.0:
            return 0.10
        # smooth cubic ramp across the transonic window
        s = (mach - 1.2) / (4.0 - 1.2)
        return 0.10 * (3.0 * s * s - 2.0 * s * s * s)

    def lift_coefficient(self, mach: float, alpha: float) -> float:
        """C_L(M, alpha) — Eq. (14)."""
        return float(self._C_L0(alpha) + self._C_L_M(mach))

    def drag_coefficient(self, mach: float, alpha: float) -> float:
        """C_D(M, alpha) — Eq. (15), induced drag via C_L^2/(pi e AR)."""
        C_L = self.lift_coefficient(mach, alpha)
        return float(self._C_D0_offset + C_L**2 / (np.pi * self.e * self.AR))

    def lift_to_drag(self, mach: float, alpha: float) -> float:
        """Trim L/D including Monte Carlo bias."""
        return self.lift_coefficient(mach, alpha) / self.drag_coefficient(mach, alpha) + self.ld_bias

    def evaluate(
        self,
        mach: float,
        alpha: float,
        velocity: float,
        density: float,
    ) -> AeroState:
        """Full aerodynamic state at (mach, alpha, V, rho)."""
        C_L = self.lift_coefficient(mach, alpha)
        C_D = self.drag_coefficient(mach, alpha)
        q = 0.5 * density * velocity * velocity
        return AeroState(
            C_L=C_L,
            C_D=C_D,
            L_over_D=C_L / C_D + self.ld_bias,
            dynamic_pressure_Pa=float(q),
            mach=float(mach),
        )
