"""Vehicle configuration: mass properties, parachutes, retro propulsion.

Implements the multi-stage hardware of Section 5.2:
  - entry capsule rigid-body properties (inertia tensor diag(Ixx,Iyy,Izz))
  - drogue parachute (deployed at Mach 2.2, transonic stabilization)
  - main parachute (deployed at h = 8 km, terminal V_z ~ 12 m/s)
  - retro-thruster pack (fired at h = 50 m, soft touchdown V_z <= 1.5 m/s)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Parachute:
    """Disk-gap-band parachute model (quadratic drag device)."""

    name: str
    diameter_m: float
    C_D: float
    deployed: bool = False

    @property
    def reference_area_m2(self) -> float:
        return np.pi * (self.diameter_m / 2.0) ** 2

    @property
    def drag_area_m2(self) -> float:
        """C_D * S, zero when stowed."""
        return self.C_D * self.reference_area_m2 if self.deployed else 0.0

    def deploy(self) -> None:
        self.deployed = True

    def jettison(self) -> None:
        self.deployed = False


@dataclass
class RetroThrusterPack:
    """Powered-descent retro propulsion (throttleable, velocity control)."""

    max_thrust_N: float
    specific_impulse_s: float
    propellant_kg: float
    throttle_min: float = 0.30
    throttle: float = 0.0
    g0: float = 9.80665

    @property
    def mass_flow_kg_s(self) -> float:
        return self.thrust_N / (self.specific_impulse_s * self.g0)

    @property
    def thrust_N(self) -> float:
        if self.propellant_kg <= 0.0:
            return 0.0
        return self.throttle * self.max_thrust_N

    @property
    def depleted(self) -> bool:
        return self.propellant_kg <= 0.0

    def command_throttle(self, throttle: float) -> None:
        self.throttle = float(np.clip(throttle, 0.0, 1.0))

    def burn(self, dt: float) -> float:
        """Consume propellant over dt; returns actual burn time applied."""
        if self.throttle <= 0.0 or self.depleted:
            return 0.0
        mdot = self.mass_flow_kg_s
        dm = mdot * dt
        if dm >= self.propellant_kg:
            dt_actual = self.propellant_kg / mdot
            self.propellant_kg = 0.0
            return dt_actual
        self.propellant_kg -= dm
        return dt


@dataclass
class EntryVehicle:
    """Rigid entry vehicle with staging hardware.

    Default configuration represents a ~3000 kg high-mass lifting capsule
    with trim L/D ~= 0.35 (Section 5.1).
    """

    dry_mass_kg: float = 2850.0
    inertia_diag_kg_m2: tuple[float, float, float] = (4200.0, 4800.0, 1600.0)
    reference_area_m2: float = 35.0
    drogue: Parachute = field(
        default_factory=lambda: Parachute("drogue", diameter_m=7.0, C_D=0.55)
    )
    main: Parachute = field(
        default_factory=lambda: Parachute("main", diameter_m=35.0, C_D=0.75)
    )
    retro: RetroThrusterPack = field(
        default_factory=lambda: RetroThrusterPack(
            max_thrust_N=95_000.0, specific_impulse_s=280.0, propellant_kg=260.0
        )
    )

    # ------------------------------------------------------------------
    @property
    def mass_kg(self) -> float:
        return self.dry_mass_kg + self.retro.propellant_kg

    @property
    def inertia_tensor(self) -> np.ndarray:
        return np.diag(self.inertia_diag_kg_m2)

    @property
    def inertia_inverse(self) -> np.ndarray:
        return np.diag(1.0 / np.asarray(self.inertia_diag_kg_m2, dtype=float))

    @property
    def total_drag_area_m2(self) -> float:
        """Combined parachute drag area (excludes capsule aero)."""
        return self.drogue.drag_area_m2 + self.main.drag_area_m2
