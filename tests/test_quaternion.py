"""Quaternion algebra and attitude kinetics tests (Section 2.3, Eq. 4)."""

from __future__ import annotations

import numpy as np
import pytest

from a6dof_edl.core.integrators import rk4_step
from a6dof_edl.core.quaternion import (
    dcm_to_quat,
    euler_to_quat,
    normalize,
    quat_conjugate,
    quat_error,
    quat_kinematic_rhs,
    quat_multiply,
    quat_to_dcm,
    rotate_vector,
)


class TestAlgebra:
    def test_multiply_identity(self):
        q = normalize(np.array([0.9, 0.1, -0.2, 0.3]))
        e = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(quat_multiply(q, e), q, atol=1e-15)
        np.testing.assert_allclose(quat_multiply(e, q), q, atol=1e-15)

    def test_conjugate_is_inverse(self):
        q = normalize(np.array([0.5, 0.5, 0.5, 0.5]))
        qq = quat_multiply(q, quat_conjugate(q))
        np.testing.assert_allclose(qq, [1.0, 0.0, 0.0, 0.0], atol=1e-15)

    def test_dcm_round_trip(self):
        q = euler_to_quat(0.3, -0.5, 1.2)
        q2 = dcm_to_quat(quat_to_dcm(q))
        # Sign ambiguity: compare absolute rotation.
        assert abs(float(np.dot(q, q2))) == pytest.approx(1.0, abs=1e-10)

    def test_dcm_orthonormal(self):
        C = quat_to_dcm(euler_to_quat(-1.0, 0.4, 2.8))
        np.testing.assert_allclose(C @ C.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(C) == pytest.approx(1.0, abs=1e-12)

    def test_rotate_vector_matches_dcm(self):
        q = euler_to_quat(0.7, 0.1, -0.9)
        v = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(rotate_vector(q, v), quat_to_dcm(q) @ v, atol=1e-12)

    def test_rotation_preserves_norm(self):
        q = euler_to_quat(1.1, 2.2, -0.3)
        v = np.array([3.0, -1.0, 0.5])
        assert np.linalg.norm(rotate_vector(q, v)) == pytest.approx(np.linalg.norm(v), rel=1e-12)

    def test_error_quaternion_shortest_path(self):
        q1 = euler_to_quat(0.0, 0.0, 0.0)
        q2 = euler_to_quat(0.0, 0.0, 3.0)  # large yaw
        qe = quat_error(q2, q1)
        assert qe[0] >= 0.0  # shortest path selected

    def test_normalize_rejects_degenerate(self):
        with pytest.raises(ValueError):
            normalize(np.zeros(4))


class TestKinematics:
    def test_norm_preservation(self):
        """Eq. (4) must conserve ||q|| = 1 for arbitrary rates."""
        q = euler_to_quat(0.2, 0.3, -0.4)
        omega = np.array([0.3, -0.2, 0.1])
        qd = quat_kinematic_rhs(q, omega)
        assert float(np.dot(q, qd)) == pytest.approx(0.0, abs=1e-12)

    def test_pure_roll_integration(self):
        """Integrating constant body rate yields the analytic rotation."""
        p = 0.5  # rad/s about body X
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([p, 0.0, 0.0])
        t, dt = 0.0, 1e-3
        for _ in range(1000):
            q = normalize(rk4_step(lambda tt, qq: quat_kinematic_rhs(qq, omega), t, q, dt))
            t += dt
        expected = np.array([np.cos(p * t / 2), np.sin(p * t / 2), 0.0, 0.0])
        np.testing.assert_allclose(q, expected, atol=1e-9)

    def test_gimbal_lock_free_at_pole(self):
        """No singularity at 90 deg pitch (where Euler angles lock)."""
        q = euler_to_quat(0.0, np.pi / 2.0, 0.0)
        qd = quat_kinematic_rhs(q, np.array([0.1, 0.2, 0.3]))
        assert np.all(np.isfinite(qd))
