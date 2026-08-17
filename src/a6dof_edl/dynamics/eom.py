"""Six-degree-of-freedom equations of motion (Section 2).

State vector y in R^13:
    y[0:3]   r  - ECI position [m]
    y[3:6]   v  - ECI velocity [m/s]
    y[6:10]  q  - attitude quaternion q_B/I (scalar first)
    y[10:13] w  - body angular rates [p, q, r]^T [rad/s]

Translational propagation  : Eqs. (1)-(3)  (ECI, J2 gravity)
Attitude kinematics        : Eq. (4)       (quaternion, non-singular)
Attitude kinetics          : Eq. (5)       (Euler's equation, I^-1(M - w x Iw))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from a6dof_edl.core.constants import R_EARTH
from a6dof_edl.core.frames import wind_frame_basis
from a6dof_edl.core.quaternion import (
    dcm_to_quat,
    normalize,
    quat_error,
    quat_kinematic_rhs,
    quat_to_dcm,
    rotate_vector,
)
from a6dof_edl.environment.atmosphere import USStandardAtmosphere1976
from a6dof_edl.environment.gravity import J2GravityModel
from a6dof_edl.guidance.phase_manager import FlightPhase
from a6dof_edl.vehicle.aerodynamics import EntryVehicleAerodynamics
from a6dof_edl.vehicle.vehicle import EntryVehicle


@dataclass
class ForceMomentBreakdown:
    """Diagnostic decomposition of the most recent RHS evaluation."""

    F_gravity_N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    F_aero_N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    F_chute_N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    F_thrust_N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    M_control_Nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    M_aero_Nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    alpha_rad: float = 0.0
    mach: float = 0.0
    q_dyn_Pa: float = 0.0
    C_L: float = 0.0
    C_D: float = 0.0


@dataclass
class AttitudeCommand:
    """Commanded attitude inputs produced by guidance each cycle."""

    bank_angle_rad: float = 0.0
    alpha_trim_rad: float = 0.4887  # 28 deg
    nose_up_descent: bool = False   # powered descent: align +X with vertical


class AttitudePDController:
    """Rigid-body attitude controller driving q -> q_cmd.

    M = Kp * qe_vec - Kd * omega, with actuator torque saturation.
    Gains are scheduled against the diagonal inertia tensor.
    """

    def __init__(
        self,
        inertia: np.ndarray,
        max_torque_Nm: float = 8_000.0,
        zeta: float = 0.95,
        omega_n_rad_s: float = 2.0,
    ) -> None:
        self.I = inertia
        self.max_torque = float(max_torque_Nm)
        self.Kp = (omega_n_rad_s**2) * np.diag(inertia) * 1.0
        self.Kd = 2.0 * zeta * omega_n_rad_s * np.diag(inertia)

    def torque(self, q_cur: np.ndarray, omega: np.ndarray, q_cmd: np.ndarray) -> np.ndarray:
        qe = quat_error(q_cmd, q_cur)
        M = self.Kp * qe[1:] - self.Kd * omega
        return np.clip(M, -self.max_torque, self.max_torque)


class SixDOFDynamics:
    """Complete 6-DOF vehicle dynamics model.

    Wires together gravity, atmosphere, capsule aerodynamics, parachute
    drag, retro thrust, and the attitude PD controller into the state
    derivative function used by the integrator.
    """

    def __init__(
        self,
        vehicle: EntryVehicle,
        aero: EntryVehicleAerodynamics,
        atmosphere: USStandardAtmosphere1976,
        gravity: J2GravityModel,
        attitude_controller: AttitudePDController | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.aero = aero
        self.atmo = atmosphere
        self.grav = gravity
        self.att_ctl = attitude_controller or AttitudePDController(vehicle.inertia_tensor)

        self.phase = FlightPhase.HYPERSONIC_ENTRY
        self.attitude_command = AttitudeCommand()
        self.diagnostic = ForceMomentBreakdown()

    # ------------------------------------------------------------------
    def commanded_attitude(self, r: np.ndarray, v: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
        """Build the commanded body->inertial DCM from guidance inputs.

        Entry / chute phases : body X nose at trim alpha above the velocity
                               vector, banked by sigma_c about the velocity
                               axis (lift-vector modulation).
        Powered descent      : body X aligned with local vertical (nose up),
                               so retro thrust opposes the descent.
        """
        r_hat = r / np.linalg.norm(r)
        if self.attitude_command.nose_up_descent:
            # Nose straight up; choose yaw arbitrarily about the vertical.
            x_b = r_hat
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(ref, x_b)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            z_b = np.cross(x_b, ref)
            z_b /= np.linalg.norm(z_b)
            y_b = np.cross(z_b, x_b)
            C = np.column_stack([x_b, y_b, z_b])
            return dcm_to_quat(C)

        e_v, e_l, e_c = wind_frame_basis(r, v)
        sigma = self.attitude_command.bank_angle_rad
        alpha = self.attitude_command.alpha_trim_rad

        # Lift direction rotated about the velocity axis by the bank angle.
        l_dir = e_l * np.cos(sigma) + e_c * np.sin(sigma)
        # Nose (body X): alpha above the velocity vector toward lift dir.
        x_b = np.cos(alpha) * e_v + np.sin(alpha) * l_dir
        x_b /= np.linalg.norm(x_b)
        # Body Z (down): perpendicular to X in the lift plane, pointing away
        # from the lift direction; body Y completes the triad.
        z_b = -(np.cos(alpha) * l_dir - np.sin(alpha) * e_v)
        z_b = z_b - np.dot(z_b, x_b) * x_b
        z_b /= np.linalg.norm(z_b)
        y_b = np.cross(z_b, x_b)
        C = np.column_stack([x_b, y_b, z_b])
        return dcm_to_quat(C)

    # ------------------------------------------------------------------
    def _angle_of_attack(self, r: np.ndarray, v: np.ndarray, q: np.ndarray) -> float:
        """Trim alpha from body X axis relative to the velocity vector."""
        v_hat = v / np.linalg.norm(v)
        x_b = rotate_vector(q, np.array([1.0, 0.0, 0.0]))
        return float(np.arccos(np.clip(np.dot(x_b, v_hat), -1.0, 1.0)))

    # ------------------------------------------------------------------
    def aerodynamic_forces(
        self, r: np.ndarray, v: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve capsule + parachute aerodynamic forces in ECI.

        Lift acts perpendicular to the velocity in the body symmetry plane
        (its direction therefore follows the banked attitude); drag opposes
        velocity; parachutes contribute pure drag along -v_hat.
        """
        altitude = np.linalg.norm(r) - R_EARTH
        speed = np.linalg.norm(v)
        atm = self.atmo.properties(max(altitude, 0.0))
        mach = speed / atm.speed_of_sound_m_s
        alpha = self._angle_of_attack(r, v, q)

        aero = self.aero.evaluate(mach, alpha, speed, atm.density_kg_m3)
        q_dyn = aero.dynamic_pressure_Pa

        # Wind-frame basis in ECI; lift direction from actual body attitude.
        e_v, _, _ = wind_frame_basis(r, v)
        C_BI = quat_to_dcm(q)
        z_body_inertial = C_BI @ np.array([0.0, 0.0, 1.0])
        # Lift along negative body-Z projected perpendicular to velocity.
        l_vec = -z_body_inertial
        l_perp = l_vec - np.dot(l_vec, e_v) * e_v
        n = np.linalg.norm(l_perp)
        l_hat = l_perp / n if n > 1e-9 else np.cross(e_v, np.array([0.0, 0.0, 1.0]))

        L = q_dyn * self.vehicle.reference_area_m2 * aero.C_L
        D = q_dyn * self.vehicle.reference_area_m2 * aero.C_D
        F_aero = L * l_hat - D * e_v

        # Parachute drag (drogue and/or main when deployed).
        D_chute = q_dyn * self.vehicle.total_drag_area_m2
        F_chute = -D_chute * e_v

        self.diagnostic.alpha_rad = alpha
        self.diagnostic.mach = mach
        self.diagnostic.q_dyn_Pa = q_dyn
        self.diagnostic.C_L = aero.C_L
        self.diagnostic.C_D = aero.C_D
        self.diagnostic.F_aero_N = F_aero
        self.diagnostic.F_chute_N = F_chute
        return F_aero + F_chute, e_v

    # ------------------------------------------------------------------
    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Full 13-state derivative (Eqs. 1-5)."""
        r = y[0:3]
        v = y[3:6]
        q = normalize(y[6:10])
        omega = y[10:13]

        m = self.vehicle.mass_kg

        # --- Translational dynamics (Eqs. 1-2) --------------------------
        F_g = m * self.grav.acceleration(r)
        F_aero, _ = self.aerodynamic_forces(r, v, q)

        F_thrust = np.zeros(3)
        if self.phase == FlightPhase.POWERED_DESCENT and self.vehicle.retro.throttle > 0.0:
            # Thrust along body +X (nose-up during descent).
            thrust_dir = rotate_vector(q, np.array([1.0, 0.0, 0.0]))
            F_thrust = self.vehicle.retro.thrust_N * thrust_dir

        v_dot = (F_aero + F_thrust) / m + F_g / m

        # --- Attitude kinematics (Eq. 4) --------------------------------
        q_dot = quat_kinematic_rhs(q, omega)

        # --- Attitude kinetics (Eq. 5) ----------------------------------
        q_cmd = self.commanded_attitude(r, v, q)
        M_ctl = self.att_ctl.torque(q, omega, q_cmd)

        # Aerodynamic damping moment (weathervane stability at trim).
        # Reuse the density/dynamic-pressure from the force evaluation.
        speed = np.linalg.norm(v)
        q_dyn = self.diagnostic.q_dyn_Pa
        c_damp = (q_dyn / max(speed, 1.0)) * self.vehicle.reference_area_m2 * 2.0
        M_aero = -c_damp * omega

        I = self.vehicle.inertia_tensor
        I_inv = self.vehicle.inertia_inverse
        M_total = M_ctl + M_aero
        omega_dot = I_inv @ (M_total - np.cross(omega, I @ omega))

        self.diagnostic.F_gravity_N = F_g
        self.diagnostic.F_thrust_N = F_thrust
        self.diagnostic.M_control_Nm = M_ctl
        self.diagnostic.M_aero_Nm = M_aero

        return np.concatenate((v, v_dot, q_dot, omega_dot))
