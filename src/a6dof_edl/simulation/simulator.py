"""Top-level EDL simulation driver.

Runs the full 6-DOF propagation from entry interface (h = 120 km,
V ~ 7.5 km/s, gamma >= -5.5 deg) through hypersonic bank-guided entry,
drogue and main parachute descent, and the powered retro-burn to
touchdown, per the multi-stage sequence of Section 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from a6dof_edl.core.constants import DEG2RAD, R_EARTH
from a6dof_edl.core.frames import ecef_to_eci, flight_path_angle
from a6dof_edl.core.integrators import rk4_step
from a6dof_edl.core.quaternion import normalize
from a6dof_edl.dynamics.eom import AttitudeCommand, SixDOFDynamics
from a6dof_edl.environment.atmosphere import USStandardAtmosphere1976
from a6dof_edl.environment.gravity import J2GravityModel
from a6dof_edl.guidance.bank_guidance import ApolloBankGuidance, ReferenceProfile
from a6dof_edl.guidance.phase_manager import FlightPhase, PhaseManager, PhaseTriggers
from a6dof_edl.guidance.touchdown_control import RetroDescentController
from a6dof_edl.simulation.results import EDLTrajectory, TrajectorySample
from a6dof_edl.vehicle.aerodynamics import EntryVehicleAerodynamics
from a6dof_edl.vehicle.vehicle import EntryVehicle


@dataclass
class EntryInterfaceState:
    """Entry interface definition (Section 1: V_E ~ 7.5 km/s, h = 120 km)."""

    altitude_m: float = 120_000.0
    velocity_m_s: float = 7_500.0
    flight_path_angle_deg: float = -5.5
    latitude_deg: float = 21.0
    longitude_deg: float = -77.5
    heading_deg: float = 90.0  # due east

    def to_eci(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the ECI state from the entry interface definition."""
        lat = self.latitude_deg * DEG2RAD
        lon = self.longitude_deg * DEG2RAD
        gamma = self.flight_path_angle_deg * DEG2RAD
        heading = self.heading_deg * DEG2RAD

        r_mag = R_EARTH + self.altitude_m
        r_hat = np.array([np.cos(lat) * np.cos(lon),
                          np.cos(lat) * np.sin(lon),
                          np.sin(lat)])
        r = r_mag * r_hat

        # Local ENU basis at the entry point.
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.array([-np.sin(lat) * np.cos(lon),
                          -np.sin(lat) * np.sin(lon),
                          np.cos(lat)])
        up = r_hat
        v_dir = (np.cos(gamma) * (np.cos(heading) * north + np.sin(heading) * east)
                 + np.sin(gamma) * up)
        v = self.velocity_m_s * v_dir / np.linalg.norm(v_dir)
        return r, v


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""

    entry: EntryInterfaceState = field(default_factory=EntryInterfaceState)
    dt_s: float = 0.05
    t_max_s: float = 4_000.0
    density_scale: float = 1.0
    ld_bias: float = 0.0
    mass_scale: float = 1.0
    guidance_enabled: bool = True
    fixed_bank_deg: float | None = None  # e.g. 0.0 for unbanked baseline run
    log_decimation: int = 4
    # Phase-adaptive integration step sizes [s]: tight where the dynamics
    # are stiff (entry pull-up, powered descent), relaxed on the chutes.
    dt_entry_s: float = 0.05
    dt_drogue_s: float = 0.10
    dt_main_s: float = 0.20
    dt_powered_s: float = 0.02


class EDLSimulator:
    """Closed-loop 6-DOF EDL simulator.

    Each integration cycle:
      1. environment + aerodynamic evaluation
      2. guidance (bank command or phase-specific controller)
      3. attitude command build
      4. RK4 propagation of the 13-element state
      5. phase sequencing and event logging
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        cfg = self.config

        self.vehicle = EntryVehicle(dry_mass_kg=2850.0 * cfg.mass_scale)
        self.aero = EntryVehicleAerodynamics(
            reference_area_m2=self.vehicle.reference_area_m2,
            ld_bias=cfg.ld_bias,
        )
        self.atmo = USStandardAtmosphere1976(density_scale=cfg.density_scale)
        self.grav = J2GravityModel()
        self.dynamics = SixDOFDynamics(self.vehicle, self.aero, self.atmo, self.grav)

        v0 = cfg.entry.velocity_m_s
        self.guidance = ApolloBankGuidance(
            reference=ReferenceProfile(v_entry=v0),
        )
        self.retro_ctl = RetroDescentController()
        self.phases = PhaseManager(PhaseTriggers())

    # ------------------------------------------------------------------
    def _initial_state(self) -> np.ndarray:
        r, v = self.config.entry.to_eci()
        # Initialize attitude at commanded trim (zero initial rates).
        self.dynamics.attitude_command = AttitudeCommand(bank_angle_rad=0.0)
        q0 = self.dynamics.commanded_attitude(r, v, np.array([1.0, 0.0, 0.0, 0.0]))
        return np.concatenate((r, v, q0, np.zeros(3)))

    # ------------------------------------------------------------------
    def _guidance_cycle(self, t: float, y: np.ndarray) -> None:
        """Compute and apply guidance commands for the current phase."""
        r, v, q = y[0:3], y[3:6], y[6:10]
        altitude = np.linalg.norm(r) - R_EARTH
        atm = self.atmo.properties(max(altitude, 0.0))
        speed = np.linalg.norm(v)
        mach = speed / atm.speed_of_sound_m_s
        alpha = self.dynamics.diagnostic.alpha_rad or self.aero.alpha_trim
        aero_state = self.aero.evaluate(mach, alpha, speed, atm.density_kg_m3)
        g_local = self.grav.magnitude(r)

        cmd = AttitudeCommand()

        if self.phases.bank_guidance_active:
            if not self.config.guidance_enabled and self.config.fixed_bank_deg is not None:
                sigma = self.config.fixed_bank_deg * DEG2RAD
            else:
                gc = self.guidance.compute(
                    r, v,
                    self.vehicle.mass_kg,
                    aero_state.C_L, aero_state.C_D,
                    self.vehicle.reference_area_m2,
                    atm.density_kg_m3, aero_state.dynamic_pressure_Pa,
                    g_local,
                )
                sigma = gc.bank_angle_rad
            cmd.bank_angle_rad = sigma
            cmd.alpha_trim_rad = self.aero.alpha_trim
            cmd.nose_up_descent = False

        elif self.phases.phase == FlightPhase.DROGUE_DESCENT:
            cmd.bank_angle_rad = 0.0
            cmd.alpha_trim_rad = self.aero.alpha_trim
            cmd.nose_up_descent = False

        elif self.phases.phase == FlightPhase.MAIN_CHUTE_DESCENT:
            # Pre-slew to nose-up under the main chute so the retro pack
            # fires along the vertical the instant powered descent starts.
            cmd.nose_up_descent = True
            cmd.alpha_trim_rad = 0.0

        elif self.phases.phase == FlightPhase.POWERED_DESCENT:
            # Vertical velocity from radial component of ECI velocity.
            r_hat = r / np.linalg.norm(r)
            v_vert = float(np.dot(v, r_hat))
            throttle = self.retro_ctl.throttle_command(
                altitude, v_vert, self.vehicle.mass_kg, g_local,
                self.vehicle.retro.max_thrust_N,
            )
            self.vehicle.retro.command_throttle(throttle)
            cmd.nose_up_descent = True
            cmd.alpha_trim_rad = 0.0

        self.dynamics.attitude_command = cmd
        self._last_mach = mach
        self._last_q_dyn = aero_state.dynamic_pressure_Pa

    # ------------------------------------------------------------------
    def _apply_staging(self) -> None:
        """Fire hardware actions bound to phase transitions."""
        phase = self.phases.phase
        if phase == FlightPhase.DROGUE_DESCENT and not self.vehicle.drogue.deployed:
            self.vehicle.drogue.deploy()
        elif phase == FlightPhase.MAIN_CHUTE_DESCENT and not self.vehicle.main.deployed:
            self.vehicle.drogue.jettison()
            self.vehicle.main.deploy()
        elif phase == FlightPhase.POWERED_DESCENT and self.vehicle.main.deployed:
            self.vehicle.main.jettison()

    # ------------------------------------------------------------------
    def run(self) -> EDLTrajectory:
        """Propagate from entry interface to touchdown; return trajectory."""
        cfg = self.config
        y = self._initial_state()
        t = 0.0
        dt = cfg.dt_entry_s
        samples: list[TrajectorySample] = []

        self.dynamics.phase = self.phases.phase

        step = 0
        while t < cfg.t_max_s:
            # Guidance and staging at the top of each cycle.
            self._guidance_cycle(t, y)
            self._apply_staging()
            self.dynamics.phase = self.phases.phase

            if step % cfg.log_decimation == 0:
                samples.append(self._sample(t, y))

            y = rk4_step(self.dynamics.rhs, t, y, dt)
            y[6:10] = normalize(y[6:10])  # guard quaternion drift

            # Propellant bookkeeping (mass properties change during burn).
            if self.dynamics.phase == FlightPhase.POWERED_DESCENT:
                self.vehicle.retro.burn(dt)

            t += dt
            step += 1

            altitude = np.linalg.norm(y[0:3]) - R_EARTH
            speed = np.linalg.norm(y[3:6])
            if self.phases.update(t, altitude, self._last_mach, speed):
                self.dynamics.phase = self.phases.phase
                dt = {
                    FlightPhase.HYPERSONIC_ENTRY: cfg.dt_entry_s,
                    FlightPhase.DROGUE_DESCENT: cfg.dt_drogue_s,
                    FlightPhase.MAIN_CHUTE_DESCENT: cfg.dt_main_s,
                    FlightPhase.POWERED_DESCENT: cfg.dt_powered_s,
                }.get(self.phases.phase, dt)
                if self.phases.terminated:
                    samples.append(self._sample(t, y))
                    break

            # Safety stop: impact without powered descent should not loop.
            if altitude <= 0.0 and not self.phases.terminated:
                self.phases._advance(t, FlightPhase.TOUCHDOWN, altitude,
                                     self._last_mach, speed)
                samples.append(self._sample(t, y))
                break

        return EDLTrajectory(
            samples=samples,
            transitions=self.phases.history,
            config_summary={
                "entry": vars(cfg.entry),
                "density_scale": cfg.density_scale,
                "ld_bias": cfg.ld_bias,
                "mass_scale": cfg.mass_scale,
                "guidance_enabled": cfg.guidance_enabled,
                "fixed_bank_deg": cfg.fixed_bank_deg,
            },
        )

    # ------------------------------------------------------------------
    def _sample(self, t: float, y: np.ndarray) -> TrajectorySample:
        r, v, q, w = y[0:3], y[3:6], y[6:10], y[10:13]
        diag = self.dynamics.diagnostic
        altitude = np.linalg.norm(r) - R_EARTH
        r_hat = r / np.linalg.norm(r)
        speed = float(np.linalg.norm(v))
        gamma = flight_path_angle(r, v)
        v_vert = float(np.dot(v, r_hat))

        # Vertical acceleration h_ddot via Eq. (10) with current bank.
        sigma = self.dynamics.attitude_command.bank_angle_rad
        g_local = self.grav.magnitude(r)
        r_mag = np.linalg.norm(r)
        m = self.vehicle.mass_kg
        aero_factor = diag.q_dyn_Pa * self.vehicle.reference_area_m2 / m
        h_ddot = ApolloBankGuidance.vertical_acceleration(
            aero_factor * diag.C_D, aero_factor * diag.C_L,
            gamma, sigma, g_local, speed, r_mag,
        )

        return TrajectorySample(
            t_s=t,
            r_eci=r.copy(),
            v_eci=v.copy(),
            q_bi=normalize(q.copy()),
            omega_body=w.copy(),
            altitude_m=float(altitude),
            speed_m_s=speed,
            mach=diag.mach,
            dynamic_pressure_Pa=diag.q_dyn_Pa,
            flight_path_angle_rad=float(gamma),
            vertical_speed_m_s=v_vert,
            vertical_accel_m_s2=float(h_ddot),
            bank_angle_rad=float(sigma),
            alpha_rad=float(diag.alpha_rad),
            C_L=diag.C_L,
            C_D=diag.C_D,
            mass_kg=float(m),
            phase=self.phases.phase,
        )
