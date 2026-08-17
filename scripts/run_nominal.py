#!/usr/bin/env python3
"""Run the nominal closed-loop EDL mission and print Section-6 metrics."""

from __future__ import annotations

import sys

from a6dof_edl.simulation.simulator import EDLSimulator, SimulationConfig


def main() -> int:
    sim = EDLSimulator(SimulationConfig())
    traj = sim.run()
    m = traj.analyze()

    print("=" * 64)
    print("A6DOF-EDL :: Nominal Closed-Loop Mission")
    print("=" * 64)
    print(f"  Total flight time          : {m.total_flight_time_s:10.1f} s")
    print(f"  Peak dynamic pressure      : {m.peak_dynamic_pressure_kPa:10.2f} kPa")
    print(f"  Max vertical accel (entry) : {m.max_vertical_accel_m_s2:10.3f} m/s^2")
    print(f"  Skip-out occurred          : {str(m.skip_out_occurred):>10s}")
    print(f"  Reached powered descent    : {str(m.reached_powered_descent):>10s}")
    print(f"  Touchdown |V_z|            : {m.touchdown_vertical_speed_m_s:10.3f} m/s")
    print(f"  Touchdown total speed      : {m.touchdown_total_speed_m_s:10.3f} m/s")
    print(f"  Downrange                  : {m.downrange_km:10.1f} km")
    print("-" * 64)
    print("  Phase transitions:")
    for tr in traj.transitions:
        print(f"    t={tr.t_s:8.1f}s  {tr.from_phase.name:>18s} -> {tr.to_phase.name:<18s}"
              f"  h={tr.altitude_m:9.1f} m  M={tr.mach:5.2f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
