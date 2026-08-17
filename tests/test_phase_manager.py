"""Phase sequencing tests (Section 5.2)."""

from __future__ import annotations

import pytest

from a6dof_edl.guidance.phase_manager import FlightPhase, PhaseManager, PhaseTriggers


@pytest.fixture()
def pm() -> PhaseManager:
    return PhaseManager(PhaseTriggers())


class TestTransitions:
    def test_full_sequence_order(self, pm):
        t = 1.0
        assert pm.update(t, 25_000.0, mach=2.1, velocity_m_s=700.0)
        assert pm.phase == FlightPhase.DROGUE_DESCENT
        t += 10.0
        assert pm.update(t, 7_900.0, mach=0.4, velocity_m_s=130.0)
        assert pm.phase == FlightPhase.MAIN_CHUTE_DESCENT
        t += 10.0
        assert pm.update(t, 49.0, mach=0.05, velocity_m_s=12.0)
        assert pm.phase == FlightPhase.POWERED_DESCENT
        t += 5.0
        assert pm.update(t, -0.1, mach=0.01, velocity_m_s=1.0)
        assert pm.phase == FlightPhase.TOUCHDOWN
        assert pm.terminated
        assert len(pm.history) == 4

    def test_drogue_not_triggered_above_mach(self, pm):
        assert not pm.update(10.0, 25_000.0, mach=2.5, velocity_m_s=800.0)
        assert pm.phase == FlightPhase.HYPERSONIC_ENTRY

    def test_min_time_in_phase_guard(self, pm):
        """A single step must not cascade through multiple stages."""
        assert pm.update(1.0, 25_000.0, mach=2.1, velocity_m_s=700.0)
        # Immediately below main-chute altitude: guard must block cascade.
        assert not pm.update(1.1, 7_900.0, mach=0.4, velocity_m_s=130.0)
        assert pm.phase == FlightPhase.DROGUE_DESCENT

    def test_transition_records_state(self, pm):
        pm.update(5.0, 24_000.0, mach=2.1, velocity_m_s=690.0)
        tr = pm.history[0]
        assert tr.t_s == 5.0
        assert tr.altitude_m == 24_000.0
        assert tr.mach == pytest.approx(2.1)

    def test_guidance_activity_flags(self, pm):
        assert pm.bank_guidance_active
        assert not pm.retro_guidance_active
        pm.update(1.0, 25_000.0, 2.1, 700.0)
        assert not pm.bank_guidance_active
