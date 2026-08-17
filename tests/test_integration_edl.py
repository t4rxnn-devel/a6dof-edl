"""Full-mission integration tests (Section 6 verification).

Runs the complete closed-loop 6-DOF EDL mission and verifies every
performance claim of the specification:
  - total atmospheric skip suppression
  - h_ddot <= 0 enforced throughout the skip-risk regime
  - bounded peak dynamic pressure
  - ordered multi-stage transitions
  - soft touchdown (V_z <= 1.5 m/s)
  - quaternion normalization over the full propagation
"""

from __future__ import annotations

import numpy as np
import pytest

from a6dof_edl.guidance.phase_manager import FlightPhase
from a6dof_edl.simulation.simulator import EDLSimulator, SimulationConfig


@pytest.fixture(scope="module")
def nominal():
    sim = EDLSimulator(SimulationConfig())
    return sim.run()


@pytest.fixture(scope="module")
def unbanked():
    """Baseline: guidance disabled, fixed sigma = 0 (paper's skip case)."""
    cfg = SimulationConfig(guidance_enabled=False, fixed_bank_deg=0.0)
    return EDLSimulator(cfg).run()


@pytest.mark.integration
class TestNominalMission:
    def test_no_skip_out(self, nominal):
        assert not nominal.analyze().skip_out_occurred

    def test_vertical_acceleration_suppressed_in_skip_regime(self, nominal):
        """h_ddot <= 0 while V > 2500 m/s (the only regime where the
        vehicle carries enough energy to skip back out)."""
        for s in nominal.samples:
            if s.phase == FlightPhase.HYPERSONIC_ENTRY and s.speed_m_s > 2500.0:
                assert s.vertical_accel_m_s2 <= 1e-6, (
                    f"h_ddot = {s.vertical_accel_m_s2} at t={s.t_s:.1f}s"
                )

    def test_peak_dynamic_pressure_bounded(self, nominal):
        assert nominal.analyze().peak_dynamic_pressure_kPa <= 38.5

    def test_phase_sequence_complete_and_ordered(self, nominal):
        seq = [tr.to_phase for tr in nominal.transitions]
        assert seq == [
            FlightPhase.DROGUE_DESCENT,
            FlightPhase.MAIN_CHUTE_DESCENT,
            FlightPhase.POWERED_DESCENT,
            FlightPhase.TOUCHDOWN,
        ]

    def test_drogue_trigger_mach(self, nominal):
        tr = nominal.transitions[0]
        assert tr.mach == pytest.approx(2.2, abs=0.15)

    def test_main_trigger_altitude(self, nominal):
        tr = nominal.transitions[1]
        assert tr.altitude_m == pytest.approx(8_000.0, abs=200.0)

    def test_retro_trigger_altitude(self, nominal):
        tr = nominal.transitions[2]
        assert tr.altitude_m == pytest.approx(50.0, abs=5.0)

    def test_soft_touchdown(self, nominal):
        m = nominal.analyze()
        assert m.touchdown_vertical_speed_m_s <= 1.5
        assert m.touchdown_total_speed_m_s <= 2.5

    def test_altitude_monotonic_descent_overall(self, nominal):
        """No large-scale altitude regain anywhere (skip signature)."""
        alt = nominal.altitude
        d = np.diff(alt)
        regain = np.cumsum(np.clip(d, 0.0, None))
        # Allow integrator-scale jitter only; no sustained climb.
        assert regain.max() < 500.0

    def test_quaternion_stays_normalized(self, nominal):
        for s in nominal.samples[::50]:
            assert np.linalg.norm(s.q_bi) == pytest.approx(1.0, abs=1e-6)

    def test_lift_down_inversion_occurred(self, nominal):
        """The controller must command sigma > 90 deg during peak q."""
        banks = nominal.bank_deg
        q = nominal.q_dyn
        assert banks[q.argmax()] > 90.0

    def test_touchdown_altitude(self, nominal):
        assert abs(nominal.samples[-1].altitude_m) < 2.0


@pytest.mark.integration
class TestUnbankedBaseline:
    def test_unbanked_run_shows_lift_excess(self, unbanked):
        """With sigma = 0 the lift-driven h_ddot spike must appear —
        the failure mode the guidance exists to suppress."""
        hdd = unbanked.vertical_accel
        entry = [i for i, s in enumerate(unbanked.samples)
                 if s.phase == FlightPhase.HYPERSONIC_ENTRY]
        assert hdd[entry].max() > 5.0  # strong positive spike

    def test_guided_outperforms_unbanked_skip_metric(self, nominal, unbanked):
        e_nom = [s.vertical_accel_m_s2 for s in nominal.samples
                 if s.phase == FlightPhase.HYPERSONIC_ENTRY and s.speed_m_s > 3000.0]
        e_unb = [s.vertical_accel_m_s2 for s in unbanked.samples
                 if s.phase == FlightPhase.HYPERSONIC_ENTRY and s.speed_m_s > 3000.0]
        assert max(e_nom) <= 1e-6 < max(e_unb)
