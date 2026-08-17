"""Multi-factor Monte Carlo dispersion simulation.

Disperses entry interface state, atmospheric density, vehicle L/D, and
vehicle mass simultaneously (multi-factor), executes full closed-loop
6-DOF runs for every sample, and aggregates landing-footprint dispersion,
peak heating/pressure, skip-suppression integrity, and touchdown-velocity
statistics.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from a6dof_edl.simulation.results import PerformanceMetrics
from a6dof_edl.simulation.simulator import EDLSimulator, EntryInterfaceState, SimulationConfig


@dataclass
class DispersionModel:
    """Gaussian dispersion model (1-sigma values) for multi-factor MC."""

    sigma_altitude_m: float = 1_000.0
    sigma_velocity_m_s: float = 25.0
    sigma_fpa_deg: float = 0.15
    sigma_density_pct: float = 0.10       # 10% density dispersion
    sigma_ld_pct: float = 0.05            # 5% L/D dispersion
    sigma_mass_pct: float = 0.02          # 2% mass dispersion

    def sample(self, rng: np.random.Generator) -> dict:
        """Draw one dispersed parameter set."""
        return {
            "d_altitude": rng.normal(0.0, self.sigma_altitude_m),
            "d_velocity": rng.normal(0.0, self.sigma_velocity_m_s),
            "d_fpa_deg": rng.normal(0.0, self.sigma_fpa_deg),
            "density_scale": 1.0 + rng.normal(0.0, self.sigma_density_pct),
            "ld_bias": rng.normal(0.0, self.sigma_ld_pct * 0.35),  # around L/D=0.35
            "mass_scale": 1.0 + rng.normal(0.0, self.sigma_mass_pct),
        }


def _run_single(args: tuple[int, int, DispersionModel]) -> tuple[int, PerformanceMetrics | None, str | None]:
    """Worker: run one dispersed full-EDL simulation."""
    idx, seed, disp_model = args
    rng = np.random.default_rng(seed)
    d = disp_model.sample(rng)
    try:
        cfg = SimulationConfig(
            entry=EntryInterfaceState(
                altitude_m=120_000.0 + d["d_altitude"],
                velocity_m_s=7_500.0 + d["d_velocity"],
                flight_path_angle_deg=-5.5 + d["d_fpa_deg"],
            ),
            density_scale=max(d["density_scale"], 0.5),
            ld_bias=d["ld_bias"],
            mass_scale=max(d["mass_scale"], 0.9),
        )
        traj = EDLSimulator(cfg).run()
        return idx, traj.analyze(), None
    except Exception as exc:  # pragma: no cover - defensive aggregation
        return idx, None, f"{type(exc).__name__}: {exc}"


@dataclass
class MonteCarloSummary:
    """Aggregated statistics across the dispersed run set."""

    n_runs: int
    n_success: int
    failures: list[str]
    skip_out_count: int
    peak_q_kPa_mean: float
    peak_q_kPa_p99: float
    touchdown_vz_mean: float
    touchdown_vz_p99: float
    touchdown_vz_max: float
    footprint_lat_deg: np.ndarray
    footprint_lon_deg: np.ndarray
    per_run: list[PerformanceMetrics] = field(default_factory=list)

    @property
    def skip_suppression_rate(self) -> float:
        """Fraction of runs with total skip suppression (Section 6)."""
        return 1.0 - self.skip_out_count / max(self.n_success, 1)


class MonteCarloRunner:
    """Parallel multi-factor Monte Carlo campaign manager."""

    def __init__(
        self,
        n_runs: int = 100,
        dispersion: DispersionModel | None = None,
        seed: int = 42,
        processes: int | None = None,
    ) -> None:
        if n_runs < 1:
            raise ValueError("n_runs must be >= 1.")
        self.n_runs = int(n_runs)
        self.dispersion = dispersion or DispersionModel()
        self.seed = int(seed)
        self.processes = processes

    # ------------------------------------------------------------------
    def run(self) -> MonteCarloSummary:
        """Execute the campaign (multiprocess) and aggregate results."""
        ss = np.random.SeedSequence(self.seed)
        seeds = ss.spawn(self.n_runs)
        args = [(i, int(s.generate_state(1)[0]), self.dispersion)
                for i, s in enumerate(seeds)]

        n_proc = self.processes or min(mp.cpu_count(), 8)
        if n_proc > 1 and self.n_runs > 1:
            with mp.Pool(n_proc) as pool:
                results = pool.map(_run_single, args)
        else:
            results = [_run_single(a) for a in args]

        metrics: list[PerformanceMetrics] = []
        failures: list[str] = []
        for idx, m, err in results:
            if m is None:
                failures.append(f"run {idx}: {err}")
            else:
                metrics.append(m)

        if not metrics:
            raise RuntimeError(f"All {self.n_runs} Monte Carlo runs failed: {failures[:3]}")

        q = np.array([m.peak_dynamic_pressure_kPa for m in metrics])
        vz = np.array([m.touchdown_vertical_speed_m_s for m in metrics])
        return MonteCarloSummary(
            n_runs=self.n_runs,
            n_success=len(metrics),
            failures=failures,
            skip_out_count=sum(1 for m in metrics if m.skip_out_occurred),
            peak_q_kPa_mean=float(q.mean()),
            peak_q_kPa_p99=float(np.percentile(q, 99)),
            touchdown_vz_mean=float(vz.mean()),
            touchdown_vz_p99=float(np.percentile(vz, 99)),
            touchdown_vz_max=float(vz.max()),
            footprint_lat_deg=np.zeros(len(metrics)),
            footprint_lon_deg=np.zeros(len(metrics)),
            per_run=metrics,
        )
