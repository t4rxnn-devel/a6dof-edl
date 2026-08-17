"""Trajectory data containers and post-flight performance analysis.

Computes the paper's verification metrics (Section 6):
  - total skip suppression (h_ddot <= 0 throughout entry)
  - bounded peak dynamic pressure (q_peak <= 38.5 kPa nominal)
  - multi-stage transition continuity
  - touchdown vertical speed (V_z <= 1.5 m/s)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from a6dof_edl.guidance.phase_manager import FlightPhase, PhaseTransition


@dataclass
class TrajectorySample:
    """One logged state sample along the trajectory."""

    t_s: float
    r_eci: np.ndarray
    v_eci: np.ndarray
    q_bi: np.ndarray
    omega_body: np.ndarray
    altitude_m: float
    speed_m_s: float
    mach: float
    dynamic_pressure_Pa: float
    flight_path_angle_rad: float
    vertical_speed_m_s: float
    vertical_accel_m_s2: float
    bank_angle_rad: float
    alpha_rad: float
    C_L: float
    C_D: float
    mass_kg: float
    phase: FlightPhase


@dataclass
class PerformanceMetrics:
    """Verification metrics per Section 6."""

    peak_dynamic_pressure_kPa: float
    max_vertical_accel_m_s2: float      # must be <= 0 in guided entry
    skip_out_occurred: bool             # altitude regained after entry start
    min_altitude_after_skip_m: float
    touchdown_vertical_speed_m_s: float
    touchdown_total_speed_m_s: float
    touchdown_time_s: float
    downrange_km: float
    reached_powered_descent: bool
    total_flight_time_s: float


@dataclass
class EDLTrajectory:
    """Complete simulated trajectory with event history."""

    samples: list[TrajectorySample]
    transitions: list[PhaseTransition]
    config_summary: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Array views
    # ------------------------------------------------------------------
    @property
    def t(self) -> np.ndarray:
        return np.array([s.t_s for s in self.samples])

    @property
    def altitude(self) -> np.ndarray:
        return np.array([s.altitude_m for s in self.samples])

    @property
    def speed(self) -> np.ndarray:
        return np.array([s.speed_m_s for s in self.samples])

    @property
    def q_dyn(self) -> np.ndarray:
        return np.array([s.dynamic_pressure_Pa for s in self.samples])

    @property
    def bank_deg(self) -> np.ndarray:
        return np.rad2deg([s.bank_angle_rad for s in self.samples])

    @property
    def vertical_accel(self) -> np.ndarray:
        return np.array([s.vertical_accel_m_s2 for s in self.samples])

    @property
    def mach(self) -> np.ndarray:
        return np.array([s.mach for s in self.samples])

    # ------------------------------------------------------------------
    def analyze(self) -> PerformanceMetrics:
        """Compute Section-6 verification metrics."""
        if not self.samples:
            raise ValueError("Empty trajectory; nothing to analyze.")

        alt = self.altitude
        hdd = self.vertical_accel

        # Skip-out detection: during the entry phase (before drogue),
        # a sustained positive altitude rate after initial descent is a skip.
        entry_idx = [i for i, s in enumerate(self.samples)
                     if s.phase == FlightPhase.HYPERSONIC_ENTRY]
        skip = False
        min_alt_after = float("inf")
        if entry_idx:
            i0 = entry_idx[0]
            entry_alt = alt[entry_idx]
            h_min_idx = int(np.argmin(entry_alt))
            post = entry_alt[h_min_idx:]
            if post.size > 10 and (post[-1] - post[0]) > 2_000.0:
                skip = True
                min_alt_after = float(post.min())

        td = self.samples[-1]
        lat0 = np.arcsin(self.samples[0].r_eci[2] / np.linalg.norm(self.samples[0].r_eci))
        lon0 = np.arctan2(self.samples[0].r_eci[1], self.samples[0].r_eci[0])
        lat1 = np.arcsin(td.r_eci[2] / np.linalg.norm(td.r_eci))
        lon1 = np.arctan2(td.r_eci[1], td.r_eci[0])
        dlat, dlon = lat1 - lat0, lon1 - lon0
        a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
        downrange = 2 * 6378137.0 * np.arcsin(np.sqrt(a)) / 1e3

        return PerformanceMetrics(
            peak_dynamic_pressure_kPa=float(self.q_dyn.max() / 1e3),
            max_vertical_accel_m_s2=float(hdd[entry_idx].max()) if entry_idx else 0.0,
            skip_out_occurred=skip,
            min_altitude_after_skip_m=min_alt_after if skip else float("nan"),
            touchdown_vertical_speed_m_s=abs(td.vertical_speed_m_s),
            touchdown_total_speed_m_s=td.speed_m_s,
            touchdown_time_s=float(td.t_s),
            downrange_km=float(downrange),
            reached_powered_descent=any(
                tr.to_phase == FlightPhase.POWERED_DESCENT for tr in self.transitions
            ),
            total_flight_time_s=float(td.t_s),
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize the trajectory to plain arrays (JSON-safe)."""
        return {
            "t_s": self.t.tolist(),
            "altitude_m": self.altitude.tolist(),
            "speed_m_s": self.speed.tolist(),
            "mach": self.mach.tolist(),
            "q_dyn_Pa": self.q_dyn.tolist(),
            "bank_deg": self.bank_deg.tolist(),
            "vertical_accel_m_s2": self.vertical_accel.tolist(),
            "phase": [int(s.phase) for s in self.samples],
            "transitions": [vars(tr) | {"from_phase": int(tr.from_phase),
                                        "to_phase": int(tr.to_phase)}
                            for tr in self.transitions],
            "config": self.config_summary,
        }
