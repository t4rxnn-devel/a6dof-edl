"""Powered retro-burn closed-loop velocity control (Section 5.2, stage 4).

A gravity-feedforward + PD vertical-velocity tracker follows a
quadratic-to-target descent profile so the vehicle touches down with
V_z <= 1.5 m/s.
"""

from __future__ import annotations

import numpy as np


class RetroDescentController:
    """Throttle command law for the powered terminal descent.

    Descent-rate profile: v_z,target = -max(v_td, sqrt(2 a_margin h))
    giving a continuously-braked approach that arrives at the surface at
    the commanded touchdown rate.
    """

    def __init__(
        self,
        touchdown_rate_m_s: float = 1.0,
        braking_margin: float = 0.07,
        kp: float = 2.0,
        kd: float = 0.10,
        min_throttle: float = 0.30,
    ) -> None:
        self.v_td = float(touchdown_rate_m_s)
        self.braking_margin = float(braking_margin)
        self.kp = float(kp)
        self.kd = float(kd)
        self.min_throttle = float(min_throttle)

    # ------------------------------------------------------------------
    def target_descent_rate(self, altitude_m: float) -> float:
        """Reference vertical velocity (negative = descending) [m/s]."""
        profile = np.sqrt(2.0 * self.braking_margin * 9.80665 * max(altitude_m, 0.0))
        return -float(max(self.v_td, profile))

    def throttle_command(
        self,
        altitude_m: float,
        v_vertical: float,
        mass_kg: float,
        g_local: float,
        max_thrust_N: float,
    ) -> float:
        """Throttle in [0, 1] enforcing the descent profile.

        thrust = m * (g + a_cmd), a_cmd = Kp (v_err) with profile feedforward.
        """
        v_ref = self.target_descent_rate(altitude_m)
        # Falling faster than the profile (v < v_ref) demands positive
        # (upward) corrective acceleration: a_cmd = -Kp (v - v_ref).
        v_err = v_vertical - v_ref
        a_cmd = -self.kp * v_err
        thrust = mass_kg * (g_local + a_cmd)
        if thrust <= 0.0:
            return 0.0
        return float(np.clip(thrust / max_thrust_N, 0.0, 1.0))
