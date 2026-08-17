# A6DOF-EDL

**Closed-Loop Bank-Vector Guidance and Autonomous Atmospheric Skip-Out Suppression for Planetary Entry Vehicles**

A fully integrated, non-singular 6-Degree-of-Freedom (6-DOF) translational and attitude dynamics flight framework with closed-loop lift-vector bank modulation, implementing the GNC specification technical report *"A6DOF-EDL"* (August 2026).

The framework eliminates atmospheric skip-out for high-mass lifting entry vehicles flying shallow corridors (γ ≥ −5.5°, trim L/D ≈ 0.35) by dynamically modulating the bank angle σ — including full lift-vector inversion (σ = 180°) — so that vertical acceleration is strictly non-positive (ḧ ≤ 0) throughout the skip-risk regime, and flies the complete multi-stage EDL sequence down to a soft powered touchdown (V_z ≤ 1.5 m/s).

## Physics Implemented

| Subsystem | Implementation | Spec reference |
|---|---|---|
| Translational EOM | ECI propagation ṙ = v, v̇ = (F_aero + F_thrust)/m + g(r) | Eqs. (1)–(2) |
| Gravity | Central body + J₂ zonal harmonic geopotential | Eq. (3) |
| Attitude kinematics | Normalized unit quaternions (non-singular, no gimbal lock) | Eq. (4) |
| Attitude kinetics | Euler rigid-body equation ω̇ = I⁻¹(M − ω × Iω), PD attitude control | Eq. (5) |
| Atmosphere | Piecewise-continuous US Standard Atmosphere 1976, 8 layers, geopotential altitude | Eqs. (6)–(9), Table 1 |
| Skip dynamics | Exact vertical acceleration ḧ(σ) | Eq. (10) |
| Skip constraint | (L/m)·cos σ ≤ g − V²/r governor | Eq. (11) |
| Entry guidance | Apollo/Orion-derived vertical-lift-fraction tracking with drag-corridor and ḣ damping feedback | Eq. (12) |
| Bank command | Saturated arccos law, lift-down inversion regime above q_dyn = 10 kPa | Eq. (13), §4.2 |
| Aerodynamics | Mach/α-dependent C_L(M,α), C_D(M,α) = C_D0 + C_L²/(πeAR), trim α = 28° → L/D ≈ 0.35 | Eqs. (14)–(15) |
| EDL sequence | Hypersonic entry → drogue (M 2.2) → main chute (8 km) → retro-burn (50 m) → touchdown | §5.2 |

## Repository Layout

```
a6dof-edl/
├── pyproject.toml
├── README.md
├── LICENSE
├── scripts/
│   ├── run_nominal.py          # Nominal closed-loop mission + Section-6 metrics
│   └── run_monte_carlo.py      # Multi-factor Monte Carlo dispersion campaign
├── src/a6dof_edl/
│   ├── core/
│   │   ├── constants.py        # Earth/US76 constants, geopotential altitude
│   │   ├── frames.py           # ECI/ECEF/NED/Body frames, wind basis, γ
│   │   ├── quaternion.py       # Hamilton algebra, DCM, Eq. (4) kinematics
│   │   └── integrators.py      # Fixed-step classic RK4
│   ├── environment/
│   │   ├── atmosphere.py       # US76 piecewise model (Eqs. 6–9)
│   │   └── gravity.py          # J2 geopotential (Eq. 3)
│   ├── vehicle/
│   │   ├── aerodynamics.py     # C_L(M,α), C_D(M,α) (Eqs. 14–15)
│   │   └── vehicle.py          # Mass props, inertia, chutes, retro pack
│   ├── dynamics/
│   │   └── eom.py              # 13-state 6-DOF EOM + attitude PD controller
│   ├── guidance/
│   │   ├── bank_guidance.py    # Apollo bank guidance + ḧ≤0 governor (Eqs. 10–13)
│   │   ├── phase_manager.py    # 4-stage EDL finite state machine (§5.2)
│   │   └── touchdown_control.py# Powered-descent velocity tracking law
│   └── simulation/
│       ├── simulator.py        # Closed-loop simulation driver
│       ├── monte_carlo.py      # Multi-factor dispersed campaign (parallel)
│       └── results.py          # Trajectory containers + Section-6 metrics
└── tests/                      # 77 tests: physics validation + full-mission verification
    ├── test_atmosphere.py      # US76 table reproduction, continuity, scaling
    ├── test_gravity.py         # Inverse square, J2 asymmetry, singularity guard
    ├── test_quaternion.py      # Algebra, DCM round-trip, norm conservation
    ├── test_dynamics.py        # RK4 (orbit energy), frames, aero, guidance math, hardware
    ├── test_phase_manager.py   # Sequencing order, triggers, anti-cascade guard
    ├── test_integration_edl.py # Full mission: skip suppression, q bound, touchdown
    └── test_monte_carlo.py     # Dispersion model statistics + campaign
```

## Installation

```bash
pip install -e .            # runtime (numpy, scipy)
pip install -e ".[dev]"     # + pytest
```

## Usage

### Nominal mission

```bash
python scripts/run_nominal.py
```

Nominal results (entry: h = 120 km, V = 7.5 km/s, γ = −5.5°):

| Metric | Result | Specification |
|---|---|---|
| Skip-out | **None** | total suppression |
| max ḧ (V > 2500 m/s regime) | **≤ 0** (governor-held) | ḧ ≤ 0 enforced |
| Peak dynamic pressure | **≈ 20.5 kPa** | bounded ≤ 38.5 kPa |
| Bank command during peak q | **σ → 180°** | lift-down inversion |
| Touchdown \|V_z\| | **≈ 1.1 m/s** | ≤ 1.5 m/s |
| Phase sequence | entry → drogue → main → retro → touchdown | §5.2 |

### Monte Carlo dispersion campaign

```bash
python scripts/run_monte_carlo.py -n 100 --processes 8
```

Simultaneously disperses entry altitude/velocity/flight-path angle, atmospheric density (±10%), vehicle L/D (±5%), and vehicle mass (±2%), runs every case closed-loop in parallel, and reports skip-suppression rate, peak-q statistics, and touchdown-velocity percentiles.

### Library API

```python
from a6dof_edl import EDLSimulator, SimulationConfig, MonteCarloRunner

traj = EDLSimulator(SimulationConfig()).run()
metrics = traj.analyze()
print(metrics.peak_dynamic_pressure_kPa, metrics.touchdown_vertical_speed_m_s)

summary = MonteCarloRunner(n_runs=100, seed=42).run()
print(summary.skip_suppression_rate)
```

## Testing

```bash
pytest                          # full suite (77 tests)
pytest -m "not integration"     # fast physics/unit tests only
```

The integration tests run the entire mission twice (guided + unbanked baseline) and assert every Section-6 claim, including that the unbanked σ = 0° baseline exhibits the lift-driven ḧ > 0 skip spike that the guidance suppresses.

## References

1. NOAA, NASA, USAF, *U.S. Standard Atmosphere, 1976*, NOAA-S/T 76-1562, 1976.
2. Hoag, D. G., *Apollo Guidance, Navigation, and Control*, MIT Instrumentation Laboratory, 1969.
3. Vinh, N. X., *Optimal Trajectories in Atmospheric Flight*, Elsevier, 1981.
4. Planet, P., et al., *Orion Entry Guidance Development and Testing*, AIAA GNC, 2010.
5. Zipfel, P. H., *Modeling and Simulation of Aerospace Vehicle Dynamics*, AIAA, 2014.
