"""Fixed-step RK4 integrator with dense event hooks.

A fixed-step classic Runge-Kutta integrator is used throughout the flight
framework (rather than adaptive solvers) so that the guidance loop runs at
a deterministic, flight-software-like rate and event detection (phase
transitions, touchdown) is sample-uniform.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

RHSFunc = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(f: RHSFunc, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """One classic fourth-order Runge-Kutta step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(f: RHSFunc, t0: float, y0: np.ndarray, dt: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Integrate n_steps fixed RK4 steps; returns (t_hist, y_hist)."""
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, y0.size))
    ts[0], ys[0] = t0, y0
    t, y = t0, y0.copy()
    for i in range(1, n_steps + 1):
        y = rk4_step(f, t, y, dt)
        t += dt
        ts[i], ys[i] = t, y
    return ts, ys
