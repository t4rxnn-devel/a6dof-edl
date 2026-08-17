"""A6DOF-EDL: Closed-Loop Bank-Vector Guidance and Autonomous Atmospheric
Skip-Out Suppression for Planetary Entry Vehicles.

A fully integrated, non-singular 6-Degree-of-Freedom (6-DOF) translational and
attitude dynamics flight framework with closed-loop lift-vector bank
modulation, per the GNC specification technical report (Aug 2026).
"""

__version__ = "1.0.0"

from a6dof_edl.simulation.simulator import EDLSimulator, SimulationConfig
from a6dof_edl.simulation.monte_carlo import MonteCarloRunner, DispersionModel

__all__ = [
    "EDLSimulator",
    "SimulationConfig",
    "MonteCarloRunner",
    "DispersionModel",
    "__version__",
]
