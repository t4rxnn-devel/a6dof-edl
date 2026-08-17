"""J2 geopotential gravity model tests (Section 2.2, Eq. 3)."""

from __future__ import annotations

import numpy as np
import pytest

from a6dof_edl.core.constants import MU_EARTH, R_EARTH
from a6dof_edl.environment.gravity import J2GravityModel


@pytest.fixture(scope="module")
def grav() -> J2GravityModel:
    return J2GravityModel()


class TestCentralBody:
    def test_surface_magnitude_equator(self, grav):
        g = grav.magnitude(np.array([R_EARTH, 0.0, 0.0]))
        assert g == pytest.approx(9.80, abs=0.05)

    def test_inverse_square_scaling(self):
        g0 = J2GravityModel(enable_j2=False)
        r1 = np.array([R_EARTH, 0.0, 0.0])
        r2 = np.array([2.0 * R_EARTH, 0.0, 0.0])
        ratio = g0.magnitude(r1) / g0.magnitude(r2)
        assert ratio == pytest.approx(4.0, rel=1e-12)

    def test_direction_is_radially_inward_without_j2(self):
        g0 = J2GravityModel(enable_j2=False)
        r = np.array([1.0, 2.0, 3.0]) * 3e6
        a = g0.acceleration(r)
        cosang = np.dot(a, r) / (np.linalg.norm(a) * np.linalg.norm(r))
        assert cosang == pytest.approx(-1.0, abs=1e-12)


class TestJ2Perturbation:
    def test_equatorial_vs_polar_asymmetry(self, grav):
        """At FIXED radius, J2 deepens the equatorial potential well:
        g_eq exceeds g_pol by ~4.5*J2 (the pole's larger surface gravity
        in reality comes from its smaller geodetic radius, not J2 sign)."""
        g_eq = grav.magnitude(np.array([R_EARTH, 0.0, 0.0]))
        g_pol = grav.magnitude(np.array([0.0, 0.0, R_EARTH]))
        assert g_eq > g_pol
        assert (g_eq - g_pol) / g_eq == pytest.approx(0.0049, abs=0.002)

    def test_j2_term_decays_faster_than_central(self, grav):
        """J2 perturbation ~ 1/r^4 must shrink relative to 1/r^2 term."""
        r_lo = np.array([R_EARTH * 1.0, 0.0, 0.0])
        r_hi = np.array([R_EARTH * 4.0, 0.0, 0.0])
        g_c = J2GravityModel(enable_j2=False)
        pert_lo = np.linalg.norm(grav.acceleration(r_lo) - g_c.acceleration(r_lo)) / g_c.magnitude(r_lo)
        pert_hi = np.linalg.norm(grav.acceleration(r_hi) - g_c.acceleration(r_hi)) / g_c.magnitude(r_hi)
        assert pert_hi < pert_lo / 10.0

    def test_equatorial_symmetry(self, grav):
        """Field is axisymmetric about the pole."""
        a = grav.acceleration(np.array([R_EARTH, 0.0, 0.5e6]))
        b = grav.acceleration(np.array([0.0, R_EARTH, 0.5e6]))
        assert np.linalg.norm(a) == pytest.approx(np.linalg.norm(b), rel=1e-12)

    def test_singularity_guard(self, grav):
        with pytest.raises(ValueError):
            grav.acceleration(np.zeros(3))
