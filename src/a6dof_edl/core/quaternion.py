"""Quaternion attitude algebra (scalar-first convention, Hamilton product).

Implements the non-singular attitude kinetics of Section 2.3 (Eq. 4):
normalized unit quaternions q = [q0, q1, q2, q3]^T eliminate gimbal lock.
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    """Return the unit quaternion, guarding against degenerate norm."""
    n = np.linalg.norm(q)
    if n < 1e-15:
        raise ValueError("Degenerate quaternion norm; cannot normalize.")
    return q / n


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2, scalar-first convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Direction cosine matrix (body -> inertial) from quaternion q_B/I."""
    q0, q1, q2, q3 = normalize(q)
    return np.array([
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
    ])


def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """Shepperd's method: DCM (body->inertial) to unit quaternion."""
    tr = np.trace(C)
    candidates = np.array([
        1.0 + tr,
        1.0 + 2.0 * C[0, 0] - tr,
        1.0 + 2.0 * C[1, 1] - tr,
        1.0 + 2.0 * C[2, 2] - tr,
    ])
    i = int(np.argmax(candidates))
    q = np.empty(4)
    if i == 0:
        q[0] = 0.5 * np.sqrt(max(candidates[0], 0.0))
        q[1] = (C[2, 1] - C[1, 2]) / (4.0 * q[0])
        q[2] = (C[0, 2] - C[2, 0]) / (4.0 * q[0])
        q[3] = (C[1, 0] - C[0, 1]) / (4.0 * q[0])
    elif i == 1:
        q[1] = 0.5 * np.sqrt(max(candidates[1], 0.0))
        q[0] = (C[2, 1] - C[1, 2]) / (4.0 * q[1])
        q[2] = (C[0, 1] + C[1, 0]) / (4.0 * q[1])
        q[3] = (C[0, 2] + C[2, 0]) / (4.0 * q[1])
    elif i == 2:
        q[2] = 0.5 * np.sqrt(max(candidates[2], 0.0))
        q[0] = (C[0, 2] - C[2, 0]) / (4.0 * q[2])
        q[1] = (C[0, 1] + C[1, 0]) / (4.0 * q[2])
        q[3] = (C[1, 2] + C[2, 1]) / (4.0 * q[2])
    else:
        q[3] = 0.5 * np.sqrt(max(candidates[3], 0.0))
        q[0] = (C[1, 0] - C[0, 1]) / (4.0 * q[3])
        q[1] = (C[0, 2] + C[2, 0]) / (4.0 * q[3])
        q[2] = (C[1, 2] + C[2, 1]) / (4.0 * q[3])
    return normalize(q)


def quat_kinematic_rhs(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Quaternion kinematic differential equation (Section 2.3, Eq. 4).

    q_dot = 1/2 * Omega(omega) q, with the 4x4 skew arrangement of Eq. 4.
    """
    p, q_rate, r_ = omega_body[0], omega_body[1], omega_body[2]
    Omega = np.array([
        [0.0, -p, -q_rate, -r_],
        [p, 0.0, r_, -q_rate],
        [q_rate, -r_, 0.0, p],
        [r_, q_rate, -p, 0.0],
    ])
    return 0.5 * Omega @ q


def rotate_vector(q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """Rotate body-frame vector to inertial frame using quaternion q_B/I."""
    qn = normalize(q)
    v_q = np.concatenate(([0.0], v_body))
    return quat_multiply(quat_multiply(qn, v_q), quat_conjugate(qn))[1:]


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """3-2-1 Euler angles [rad] to quaternion (scalar-first)."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return normalize(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]))


def quat_error(q_cmd: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Shortest-path attitude error quaternion (vector part used by PD law)."""
    qe = quat_multiply(quat_conjugate(q_cur), normalize(q_cmd))
    if qe[0] < 0.0:
        qe = -qe
    return qe
