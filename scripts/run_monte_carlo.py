#!/usr/bin/env python3
"""Run the multi-factor Monte Carlo dispersion campaign."""

from __future__ import annotations

import argparse
import sys

from a6dof_edl.simulation.monte_carlo import MonteCarloRunner


def main() -> int:
    p = argparse.ArgumentParser(description="A6DOF-EDL Monte Carlo campaign")
    p.add_argument("-n", "--runs", type=int, default=50, help="number of runs")
    p.add_argument("--seed", type=int, default=42, help="master RNG seed")
    p.add_argument("--processes", type=int, default=None, help="worker processes")
    args = p.parse_args()

    mc = MonteCarloRunner(n_runs=args.runs, seed=args.seed, processes=args.processes)
    s = mc.run()

    print("=" * 64)
    print(f"A6DOF-EDL :: Monte Carlo Campaign ({s.n_runs} runs)")
    print("=" * 64)
    print(f"  Successful runs            : {s.n_success}/{s.n_runs}")
    print(f"  Skip-out count             : {s.skip_out_count}")
    print(f"  Skip suppression rate      : {100.0 * s.skip_suppression_rate:9.1f} %")
    print(f"  Peak q_dyn  mean / p99     : {s.peak_q_kPa_mean:7.2f} / {s.peak_q_kPa_p99:7.2f} kPa")
    print(f"  Touchdown Vz mean / p99    : {s.touchdown_vz_mean:7.3f} / {s.touchdown_vz_p99:7.3f} m/s")
    print(f"  Touchdown Vz max           : {s.touchdown_vz_max:7.3f} m/s")
    if s.failures:
        print("-" * 64)
        print("  Failures:")
        for f in s.failures[:10]:
            print(f"    {f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
