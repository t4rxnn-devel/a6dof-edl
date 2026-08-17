"""Reference frame utilities (Section 2.1).

Frames:
  F_I  - Earth-Centered Inertial (non-rotating)
  F_E  - Earth-Centered Earth-Fixed (rotates at omega_E)
  F_N  - Local horizontal NED, vehicle-centric
  F_B  - Body-fixed (X nose, Y right wing, Z down)

The simulation propagates the translational state in F_I (Eq. 1-2) and
attitude as quaternion q_B/I (Eq. 4). Aerodynamic forces are resolved in
the wind frame and rotated into F_I through the velocity direction.
"""

from __future__ import annotations

import numpy as np

from a6dof_edl.core.constants import OMEGA_EARTH, R_EARTH


def eci_to_ecef(r_eci: np.ndarray, t: float, omega: float = OMEGA_EARTH) -> np.ndarray:
    """Rotate ECI position into ECEF using a simple z-rotation theta = omega*t."""
    c, s = np.cos(omega * t), np.sin(omega * t)
    C = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return C @ r_eci


def ecef_to_eci(r_ecef: np.ndarray, t: float, omega: float = OMEGA_EARTH) -> np.ndarray:
    """Inverse of :func:`eci_to_ecef`."""
    c, s = np.cos(omega * t), np.sin(omega * t)
    C = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return C @ r_ecef


def ecef_to_geodetic(r_ecef: np.ndarray, radius: float = R_EARTH) -> tuple[float, float, float]:
    """Spherical geodetic (lat, lon, alt) from ECEF position.

    Returns latitude [rad], longitude [rad], altitude above spherical
    reference radius [m]. Sufficient for atmosphere/gravity lookups and
    footprint scatter statistics; an ellipsoidal Bowring solver is not
    required at entry-interface altitudes.
    """
    r = np.linalg.norm(r_ecef)
    lat = np.arcsin(np.clip(r_ecef[2] / r, -1.0, 1.0))
    lon = np.arctan2(r_ecef[1], r_ecef[0])
    return lat, lon, r - radius


def wind_frame_basis(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct orthonormal wind-frame unit vectors in ECI.

    e_v  : along the velocity vector (drag opposes this)
    e_l  : lift direction, in the r-v plane, perpendicular to e_v,
           positive toward the local vertical ("lift-up")
    e_c  : completes the right-handed triad (cross-range)
    """
    v_hat = v / np.linalg.norm(v)
    r_hat = r / np.linalg.norm(r)
    e_c = np.cross(v_hat, r_hat)
    n = np.linalg.norm(e_c)
    if n < 1e-12:  # radial flight degeneracy: pick any perpendicular
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, v_hat)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e_c = np.cross(v_hat, tmp)
        n = np.linalg.norm(e_c)
    e_c = e_c / n
    e_l = np.cross(e_c, v_hat)
    return v_hat, e_l, e_c


def flight_path_angle(r: np.ndarray, v: np.ndarray) -> float:
    """Flight path angle gamma [rad]; negative when descending."""
    r_hat = r / np.linalg.norm(r)
    v_hat = v / np.linalg.norm(v)
    return np.arcsin(np.clip(np.dot(r_hat, v_hat), -1.0, 1.0))
