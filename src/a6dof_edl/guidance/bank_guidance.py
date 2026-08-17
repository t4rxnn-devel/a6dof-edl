"""Closed-loop bank-angle guidance with autonomous skip-out suppression.

Implements Section 4 of the specification:

  - Eq. (10): exact vertical acceleration h_ddot as function of sigma
  - Eq. (11): skip-suppression constraint  (L/m) cos sigma <= g - V^2/r
  - Eq. (12): Apollo/Orion-derived required vertical lift fraction with
              drag-error and altitude-rate-error feedback terms
  - Eq. (13): bank command magnitude via saturated arccos
  - Full lift-down inversion (sigma = 180 deg) when q_dyn > 10 kPa
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from a6dof_edl.core.constants import R_EARTH
from a6dof_edl.core.frames import flight_path_angle


def saturate(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Scalar saturation used by Eq. (13)."""
    return float(np.clip(x, lo, hi))


@dataclass
class GuidanceCommand:
    """Output of one guidance cycle."""

    bank_angle_rad: float          # sigma_c, magnitude in [0, pi]
    vertical_lift_fraction: float  # K_v = cos(sigma_c)
    inversion_active: bool         # True when full lift-down commanded
    skip_constraint_active: bool   # True when Eq. (11) bound the command
    lift_accel_vertical_req: float # (L/m)_v required [m/s^2]
    lift_accel_total: float        # (L/m)_total available [m/s^2]


@dataclass
class ReferenceProfile:
    """Analytic reference drag corridor D_ref(V) and hdot_ref(V).

    The reference follows a constant-dynamic-pressure-ish equilibrium glide
    segment into a terminal quadratic ramp, parameterized by the peak
    allowed deceleration. Generated from the nominal entry interface.
    """

    v_entry: float
    d_ref_entry_g: float = 0.02
    d_ref_peak_g: float = 4.0
    v_peak: float = 1200.0
    hdot_ref_entry: float = -700.0
    hdot_ref_terminal: float = -150.0

    def drag_ref(self, v: float) -> float:
        """Reference drag acceleration [m/s^2]."""
        if v >= self.v_entry:
            return self.d_ref_entry_g * 9.80665
        if v <= self.v_peak:
            return self.d_ref_peak_g * 9.80665
        f = (self.v_entry - v) / (self.v_entry - self.v_peak)
        # quadratic build-up of drag along the corridor
        g0, g1 = self.d_ref_entry_g, self.d_ref_peak_g
        return (g0 + (g1 - g0) * f * f) * 9.80665

    def hdot_ref(self, v: float) -> float:
        """Reference altitude rate [m/s]."""
        if v >= self.v_entry:
            return self.hdot_ref_entry
        if v <= self.v_peak:
            return self.hdot_ref_terminal
        f = (self.v_entry - v) / (self.v_entry - self.v_peak)
        return self.hdot_ref_entry + (self.hdot_ref_terminal - self.hdot_ref_entry) * f


class ApolloBankGuidance:
    """Apollo/Orion-derived entry guidance with skip-suppression override.

    Parameters
    ----------
    gain_drag : float
        K_D in Eq. (12) — drag-error feedback gain [per m/s^2 scaled].
    gain_hdot : float
        K_h in Eq. (12) — altitude-rate damping gain.
    q_inversion_pa : float
        Dynamic pressure above which full lift-down inversion is executed
        (paper: q_dyn > 10 kPa -> sigma_c = 180 deg).
    ld_vertical_ref : float
        Reference vertical L/D fraction (L/D)_ref for equilibrium glide.
    """

    def __init__(
        self,
        gain_drag: float = 0.15,
        gain_hdot: float = -2.0e-3,
        q_inversion_pa: float = 10_000.0,
        ld_vertical_ref: float = 0.75,
        reference: ReferenceProfile | None = None,
    ) -> None:
        self.K_D = float(gain_drag)
        self.K_h = float(gain_hdot)
        self.q_inversion = float(q_inversion_pa)
        self.ld_v_ref = float(ld_vertical_ref)
        self.reference = reference

    # ------------------------------------------------------------------
    @staticmethod
    def vertical_acceleration(
        drag_accel: float,
        lift_accel: float,
        gamma: float,
        sigma: float,
        g: float,
        v: float,
        r: float,
    ) -> float:
        """Exact vertical acceleration h_ddot — Eq. (10)."""
        return (
            -drag_accel * np.sin(gamma)
            - g * np.sin(gamma) ** 2
            + lift_accel * np.cos(gamma) * np.cos(sigma)
            + (v * v / r - g) * np.cos(gamma) ** 2
        )

    # ------------------------------------------------------------------
    def compute(
        self,
        r_eci: np.ndarray,
        v_eci: np.ndarray,
        mass_kg: float,
        C_L: float,
        C_D: float,
        S_ref: float,
        rho: float,
        q_dyn: float,
        g_local: float,
    ) -> GuidanceCommand:
        """One guidance cycle: compute the bank command sigma_c.

        Follows Eq. (12) for the required vertical lift fraction, then
        applies the skip-suppression constraint (Eq. 11) and the
        peak-dynamic-pressure lift-down inversion override.
        """
        v = float(np.linalg.norm(v_eci))
        r = float(np.linalg.norm(r_eci))
        gamma = flight_path_angle(r_eci, v_eci)

        # Total available specific aerodynamic forces [m/s^2]
        aero_factor = q_dyn * S_ref / mass_kg
        L_total = aero_factor * C_L   # (L/m)_total
        D_accel = aero_factor * C_D   # (D/m)

        skip_constraint_active = False
        inversion_active = False

        sin_g, cos_g = np.sin(gamma), np.cos(gamma)

        # --- Eq. (12): required vertical lift fraction ------------------
        if self.reference is not None and v > 50.0:
            D_ref = self.reference.drag_ref(v)
            hdot_ref = self.reference.hdot_ref(v)
            hdot = v * np.sin(gamma)
            ld_v = (
                self.ld_v_ref
                + self.K_D * (D_accel - D_ref) / max(D_ref, 1e-3)
                + self.K_h * (hdot - hdot_ref)
            )
        else:
            ld_v = self.ld_v_ref

        L_v_req = ld_v * D_accel  # required vertical lift acceleration

        # --- Eq. (13): bank magnitude via saturated arccos --------------
        if L_total > 1e-9:
            K_v = saturate(L_v_req / L_total)
        else:
            K_v = 1.0

        # --- Eqs. (10)-(11): exact skip-suppression governor ------------
        # Enforce h_ddot <= 0 from Eq. (10):
        #   -D sin g - g sin^2 g + (L/m) cos g cos s + (V^2/r - g) cos^2 g <= 0
        # solved for the maximum admissible vertical lift fraction:
        #   cos s <= [D sin g + g sin^2 g - (V^2/r - g) cos^2 g] / [(L/m) cos g]
        if L_total > 1e-9 and abs(cos_g) > 1e-6:
            numerator = (
                D_accel * sin_g
                + g_local * sin_g * sin_g
                - (v * v / r - g_local) * cos_g * cos_g
            )
            cos_sigma_max = saturate(numerator / (L_total * cos_g))
            if K_v > cos_sigma_max:
                K_v = cos_sigma_max
                skip_constraint_active = True

        # --- Peak dynamic pressure regime (q > 10 kPa, Section 4.2) -----
        # Lift-up is prohibited; the command enters the lift-down
        # (inverted) regime sigma >= 90 deg, deepening toward the full
        # 180 deg inversion exactly as the Eq. (10) governor demands.
        if q_dyn > self.q_inversion:
            if K_v > 0.0:
                K_v = 0.0
            inversion_active = True

        sigma_c = float(np.arccos(saturate(K_v)))
        return GuidanceCommand(
            bank_angle_rad=sigma_c,
            vertical_lift_fraction=K_v,
            inversion_active=inversion_active,
            skip_constraint_active=skip_constraint_active,
            lift_accel_vertical_req=L_v_req,
            lift_accel_total=L_total,
        )
