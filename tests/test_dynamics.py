"""Dynamics subsystem tests: integrator, frames, aero, vehicle, controllers."""

from __future__ import annotations

import numpy as np
import pytest

from a6dof_edl.core.constants import DEG2RAD, MU_EARTH, R_EARTH
from a6dof_edl.core.frames import (
    ecef_to_eci,
    eci_to_ecef,
    flight_path_angle,
    wind_frame_basis,
)
from a6dof_edl.core.integrators import integrate, rk4_step
from a6dof_edl.environment.gravity import J2GravityModel
from a6dof_edl.guidance.bank_guidance import ApolloBankGuidance, ReferenceProfile, saturate
from a6dof_edl.guidance.touchdown_control import RetroDescentController
from a6dof_edl.vehicle.aerodynamics import EntryVehicleAerodynamics
from a6dof_edl.vehicle.vehicle import EntryVehicle, Parachute, RetroThrusterPack


class TestRK4:
    def test_exponential_decay(self):
        f = lambda t, y: -y
        ts, ys = integrate(f, 0.0, np.array([1.0]), 1e-3, 1000)
        assert ys[-1, 0] == pytest.approx(np.exp(-1.0), rel=1e-10)

    def test_harmonic_oscillator_energy(self):
        omega = 2.0
        def f(t, y):
            return np.array([y[1], -omega**2 * y[0]])
        ts, ys = integrate(f, 0.0, np.array([1.0, 0.0]), 1e-3, 5000)
        E = 0.5 * (ys[:, 1] ** 2 + omega**2 * ys[:, 0] ** 2)
        assert np.abs(E - E[0]).max() < 1e-8

    def test_circular_orbit_energy_conservation(self):
        """Vacuum two-body orbit: specific energy conserved by RK4."""
        grav = J2GravityModel(enable_j2=False)
        r0 = np.array([R_EARTH + 400e3, 0.0, 0.0])
        v0 = np.array([0.0, np.sqrt(MU_EARTH / np.linalg.norm(r0)), 0.0])
        y = np.concatenate([r0, v0])
        def f(t, y):
            return np.concatenate([y[3:], grav.acceleration(y[:3])])
        ts, ys = integrate(f, 0.0, y, 0.1, 10_000)
        E = 0.5 * np.sum(ys[:, 3:] ** 2, axis=1) - MU_EARTH / np.linalg.norm(ys[:, :3], axis=1)
        assert np.abs(E - E[0]).max() / abs(E[0]) < 1e-9
        # Orbit stays circular to integrator accuracy.
        radii = np.linalg.norm(ys[:, :3], axis=1)
        assert radii.min() > R_EARTH + 390e3


class TestFrames:
    def test_eci_ecef_round_trip(self):
        r = np.array([7.0e6, -1.0e6, 2.0e6])
        t = 1234.5
        np.testing.assert_allclose(ecef_to_eci(eci_to_ecef(r, t), t), r, atol=1e-6)

    def test_wind_basis_orthonormal(self):
        r = np.array([6.5e6, 0.5e6, 0.2e6])
        v = np.array([100.0, 7400.0, -300.0])
        e_v, e_l, e_c = wind_frame_basis(r, v)
        for a in (e_v, e_l, e_c):
            assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-12)
        assert np.dot(e_v, e_l) == pytest.approx(0.0, abs=1e-12)
        assert np.dot(e_v, e_c) == pytest.approx(0.0, abs=1e-12)
        assert np.dot(e_l, e_c) == pytest.approx(0.0, abs=1e-12)

    def test_flight_path_angle_sign(self):
        r = np.array([6.5e6, 0.0, 0.0])
        v_down = np.array([-100.0, 7000.0, 0.0])
        v_up = np.array([100.0, 7000.0, 0.0])
        assert flight_path_angle(r, v_down) < 0.0
        assert flight_path_angle(r, v_up) > 0.0


class TestAerodynamics:
    def test_trim_ld_hypersonic(self):
        """Section 5.1: alpha_trim = 28 deg must yield L/D ~= 0.35."""
        aero = EntryVehicleAerodynamics()
        ld = aero.lift_to_drag(20.0, aero.alpha_trim)
        assert ld == pytest.approx(0.35, abs=1e-6)

    def test_induced_drag_relation(self):
        """Eq. (15): C_D - C_D0 = C_L^2 / (pi e AR)."""
        aero = EntryVehicleAerodynamics()
        mach, alpha = 10.0, 25.0 * DEG2RAD
        C_L = aero.lift_coefficient(mach, alpha)
        C_D = aero.drag_coefficient(mach, alpha)
        C_D_ind = C_L**2 / (np.pi * aero.e * aero.AR)
        assert C_D - aero._C_D0_offset == pytest.approx(C_D_ind, rel=1e-12)

    def test_transonic_correction_bounded(self):
        aero = EntryVehicleAerodynamics()
        vals = [aero.lift_coefficient(m, aero.alpha_trim) for m in np.linspace(0.5, 8.0, 200)]
        assert all(np.isfinite(vals))
        # Hypersonic asymptote: C_L,M saturates beyond Mach 4.
        assert aero._C_L_M(4.0) == pytest.approx(aero._C_L_M(8.0))

    def test_ld_bias_dispersion(self):
        a0 = EntryVehicleAerodynamics(ld_bias=0.0)
        a1 = EntryVehicleAerodynamics(ld_bias=0.05)
        d = a1.lift_to_drag(15.0, a0.alpha_trim) - a0.lift_to_drag(15.0, a0.alpha_trim)
        assert d == pytest.approx(0.05, rel=1e-12)


class TestGuidanceMath:
    def test_saturate(self):
        assert saturate(2.0) == 1.0
        assert saturate(-2.0) == -1.0
        assert saturate(0.3) == 0.3

    def test_vertical_acceleration_equation(self):
        """Eq. (10) spot check against hand computation."""
        hdd = ApolloBankGuidance.vertical_acceleration(
            drag_accel=10.0, lift_accel=20.0, gamma=-0.1, sigma=np.pi,
            g=9.8, v=5000.0, r=6.4e6,
        )
        expected = (
            -10.0 * np.sin(-0.1) - 9.8 * np.sin(-0.1) ** 2
            + 20.0 * np.cos(-0.1) * np.cos(np.pi)
            + (5000.0**2 / 6.4e6 - 9.8) * np.cos(-0.1) ** 2
        )
        assert hdd == pytest.approx(expected, rel=1e-12)

    def test_skip_suppression_governor_enforces_nonpositive_hddot(self):
        """In the skip-risk regime the commanded bank must satisfy Eq. 10."""
        g = ApolloBankGuidance(reference=ReferenceProfile(v_entry=7500.0))
        r = np.array([R_EARTH + 60e3, 0.0, 0.0])
        v = np.array([0.0, 6000.0, -600.0])
        cmd = g.compute(r, v, 3000.0, C_L=0.30, C_D=0.86, S_ref=35.0,
                        rho=3.0e-4, q_dyn=5.4e3, g_local=9.7)
        # Re-evaluate Eq. (10) at the commanded bank angle.
        gamma = flight_path_angle(r, v)
        L_tot = cmd.lift_accel_total
        D_acc = 5.4e3 * 35.0 / 3000.0 * 0.86
        hdd = ApolloBankGuidance.vertical_acceleration(
            D_acc, L_tot, gamma, cmd.bank_angle_rad, 9.7, np.linalg.norm(v),
            np.linalg.norm(r),
        )
        assert hdd <= 1e-9

    def test_inversion_regime_at_peak_q(self):
        """q_dyn > 10 kPa must prohibit lift-up (sigma_c >= 90 deg)."""
        g = ApolloBankGuidance(reference=ReferenceProfile(v_entry=7500.0))
        r = np.array([R_EARTH + 50e3, 0.0, 0.0])
        v = np.array([0.0, 5000.0, -800.0])
        cmd = g.compute(r, v, 3000.0, C_L=0.30, C_D=0.86, S_ref=35.0,
                        rho=1.0e-3, q_dyn=12.5e3, g_local=9.7)
        assert cmd.inversion_active
        assert cmd.bank_angle_rad >= np.pi / 2.0 - 1e-12

    def test_bank_angle_within_bounds(self):
        g = ApolloBankGuidance(reference=ReferenceProfile(v_entry=7500.0))
        for q in (100.0, 2e3, 8e3, 15e3, 30e3):
            r = np.array([R_EARTH + 55e3, 0.0, 0.0])
            v = np.array([0.0, 5500.0, -500.0])
            cmd = g.compute(r, v, 3000.0, 0.30, 0.86, 35.0, 1e-3, q, 9.7)
            assert 0.0 <= cmd.bank_angle_rad <= np.pi + 1e-12


class TestRetroController:
    def test_profile_monotonic_approach(self):
        ctl = RetroDescentController()
        v_hi = ctl.target_descent_rate(50.0)
        v_lo = ctl.target_descent_rate(5.0)
        assert v_hi < v_lo <= -ctl.v_td

    def test_throttle_brakes_when_falling_fast(self):
        ctl = RetroDescentController()
        th = ctl.throttle_command(30.0, v_vertical=-12.0, mass_kg=3000.0,
                                  g_local=9.8, max_thrust_N=95e3)
        assert 0.5 < th <= 1.0

    def test_throttle_relaxes_when_slow(self):
        ctl = RetroDescentController()
        th = ctl.throttle_command(30.0, v_vertical=-2.0, mass_kg=3000.0,
                                  g_local=9.8, max_thrust_N=95e3)
        assert th < 0.4

    def test_throttle_bounded(self):
        ctl = RetroDescentController()
        th = ctl.throttle_command(50.0, v_vertical=-100.0, mass_kg=3000.0,
                                  g_local=9.8, max_thrust_N=95e3)
        assert th == 1.0


class TestVehicleHardware:
    def test_mass_budget_includes_propellant(self):
        v = EntryVehicle()
        assert v.mass_kg == pytest.approx(v.dry_mass_kg + v.retro.propellant_kg)

    def test_chute_drag_area_staging(self):
        v = EntryVehicle()
        assert v.total_drag_area_m2 == 0.0
        v.drogue.deploy()
        assert v.total_drag_area_m2 == pytest.approx(v.drogue.drag_area_m2)
        v.drogue.jettison()
        v.main.deploy()
        assert v.total_drag_area_m2 == pytest.approx(v.main.drag_area_m2)

    def test_retro_depletion(self):
        rt = RetroThrusterPack(max_thrust_N=1000.0, specific_impulse_s=100.0,
                               propellant_kg=0.5)
        rt.command_throttle(1.0)
        rt.burn(10.0)  # mdot ~ 1.02 kg/s -> depletes 0.5 kg
        assert rt.depleted
        assert rt.thrust_N == 0.0

    def test_inertia_tensor_diagonal(self):
        v = EntryVehicle()
        I = v.inertia_tensor
        assert np.allclose(I, np.diag(np.diag(I)))
        assert np.all(np.diag(I) > 0.0)
