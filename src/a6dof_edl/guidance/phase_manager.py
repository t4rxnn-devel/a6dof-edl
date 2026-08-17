"""Multi-stage EDL sequence manager (Section 5.2).

Flight phases:
  1. HYPERSONIC_ENTRY  : 120 km -> drogue trigger (Mach 2.2)
  2. DROGUE_DESCENT    : Mach 2.2 -> h = 8 km
  3. MAIN_CHUTE_DESCENT: 8 km -> h = 50 m
  4. POWERED_DESCENT   : 50 m -> touchdown
  5. TOUCHDOWN         : terminal state
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class FlightPhase(enum.IntEnum):
    HYPERSONIC_ENTRY = 0
    DROGUE_DESCENT = 1
    MAIN_CHUTE_DESCENT = 2
    POWERED_DESCENT = 3
    TOUCHDOWN = 4


@dataclass
class PhaseTriggers:
    """Sequencing triggers per Section 5.2."""

    drogue_mach: float = 2.2
    main_altitude_m: float = 8_000.0
    retro_altitude_m: float = 50.0
    touchdown_altitude_m: float = 0.0
    min_time_in_phase_s: float = 0.5


@dataclass
class PhaseTransition:
    t_s: float
    from_phase: FlightPhase
    to_phase: FlightPhase
    altitude_m: float
    mach: float
    velocity_m_s: float


@dataclass
class PhaseManager:
    """Finite state machine sequencing the four EDL stages."""

    triggers: PhaseTriggers = field(default_factory=PhaseTriggers)
    phase: FlightPhase = FlightPhase.HYPERSONIC_ENTRY
    phase_entry_time_s: float = 0.0
    history: list[PhaseTransition] = field(default_factory=list)

    # ------------------------------------------------------------------
    def _time_in_phase(self, t: float) -> float:
        return t - self.phase_entry_time_s

    def _advance(self, t: float, to_phase: FlightPhase, alt: float, mach: float, v: float) -> bool:
        self.history.append(
            PhaseTransition(
                t_s=t,
                from_phase=self.phase,
                to_phase=to_phase,
                altitude_m=alt,
                mach=mach,
                velocity_m_s=v,
            )
        )
        self.phase = to_phase
        self.phase_entry_time_s = t
        return True

    # ------------------------------------------------------------------
    def update(
        self,
        t: float,
        altitude_m: float,
        mach: float,
        velocity_m_s: float,
    ) -> bool:
        """Evaluate transition conditions; returns True if a stage fired.

        Each transition is guarded by a minimum time-in-phase so that a
        single integrator step cannot skip through multiple stages.
        """
        if self._time_in_phase(t) < self.triggers.min_time_in_phase_s:
            return False

        if self.phase == FlightPhase.HYPERSONIC_ENTRY:
            if mach <= self.triggers.drogue_mach and altitude_m < 30_000.0:
                return self._advance(t, FlightPhase.DROGUE_DESCENT, altitude_m, mach, velocity_m_s)

        elif self.phase == FlightPhase.DROGUE_DESCENT:
            if altitude_m <= self.triggers.main_altitude_m:
                return self._advance(t, FlightPhase.MAIN_CHUTE_DESCENT, altitude_m, mach, velocity_m_s)

        elif self.phase == FlightPhase.MAIN_CHUTE_DESCENT:
            if altitude_m <= self.triggers.retro_altitude_m:
                return self._advance(t, FlightPhase.POWERED_DESCENT, altitude_m, mach, velocity_m_s)

        elif self.phase == FlightPhase.POWERED_DESCENT:
            if altitude_m <= self.triggers.touchdown_altitude_m:
                return self._advance(t, FlightPhase.TOUCHDOWN, altitude_m, mach, velocity_m_s)

        return False

    # ------------------------------------------------------------------
    @property
    def bank_guidance_active(self) -> bool:
        """Bank modulation only during hypersonic entry (Section 5.2-1)."""
        return self.phase == FlightPhase.HYPERSONIC_ENTRY

    @property
    def retro_guidance_active(self) -> bool:
        return self.phase == FlightPhase.POWERED_DESCENT

    @property
    def terminated(self) -> bool:
        return self.phase == FlightPhase.TOUCHDOWN
