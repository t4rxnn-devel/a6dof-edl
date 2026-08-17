"""US Standard Atmosphere 1976 validation tests (Section 3, Table 1).

Reference values are the published US76 table entries; tolerance bands
account for the geopotential/geometric altitude conversion applied here.
"""

from __future__ import annotations

import numpy as np
import pytest

from a6dof_edl.environment.atmosphere import USStandardAtmosphere1976


@pytest.fixture(scope="module")
def atmo() -> USStandardAtmosphere1976:
    return USStandardAtmosphere1976()


class TestTableValues:
    """Base-row reproduction at layer boundaries (h_g ~= h at low alt)."""

    def test_sea_level(self, atmo):
        s = atmo.properties(0.0)
        assert s.temperature_K == pytest.approx(288.15, abs=0.5)
        assert s.pressure_Pa == pytest.approx(101325.0, rel=1e-3)
        assert s.density_kg_m3 == pytest.approx(1.2250, rel=2e-2)
        assert s.speed_of_sound_m_s == pytest.approx(340.3, abs=1.0)

    def test_11km_tropopause(self, atmo):
        s = atmo.properties(11_000.0)
        assert s.temperature_K == pytest.approx(216.65, abs=1.0)
        assert s.pressure_Pa == pytest.approx(22632.0, rel=5e-2)

    def test_20km(self, atmo):
        s = atmo.properties(20_000.0)
        assert s.temperature_K == pytest.approx(216.65, abs=1.5)
        assert s.pressure_Pa == pytest.approx(5474.9, rel=5e-2)

    def test_32km(self, atmo):
        s = atmo.properties(32_000.0)
        assert s.temperature_K == pytest.approx(228.65, abs=2.0)
        assert s.pressure_Pa == pytest.approx(868.02, rel=8e-2)

    def test_47km(self, atmo):
        s = atmo.properties(47_000.0)
        assert s.pressure_Pa == pytest.approx(110.91, rel=8e-2)

    def test_51km(self, atmo):
        s = atmo.properties(51_000.0)
        assert s.pressure_Pa == pytest.approx(66.938, rel=8e-2)

    def test_71km(self, atmo):
        # 71 km is a geopotential table altitude: h_geom = R*hg/(R-hg).
        s = atmo.properties(71_800.0)
        assert s.pressure_Pa == pytest.approx(3.9564, rel=1.5e-1)


class TestLayerPhysics:
    def test_gradient_layer_temperature_slope(self, atmo):
        """Layer 0 lapse rate -6.5 K/km (Eq. 6)."""
        t0 = atmo.temperature(1_000.0)
        t1 = atmo.temperature(2_000.0)
        assert (t1 - t0) / 1000.0 == pytest.approx(-6.5e-3, abs=1e-4)

    def test_isothermal_layer_temperature_constant(self, atmo):
        """Layer 1 (11-20 km) is isothermal at 216.65 K."""
        for h in (12_000.0, 15_000.0, 19_000.0):
            assert atmo.temperature(h) == pytest.approx(216.65, abs=1.5)

    def test_pressure_monotonically_decreasing(self, atmo):
        hs = np.linspace(0.0, 84_000.0, 500)
        ps = np.array([atmo.pressure(h) for h in hs])
        assert np.all(np.diff(ps) < 0.0)

    def test_pressure_continuity_across_layers(self, atmo):
        """Piecewise formulation must be continuous at every base altitude."""
        for hb in (11e3, 20e3, 32e3, 47e3, 51e3, 71e3):
            p_lo = atmo.pressure(hb - 1.0)
            p_hi = atmo.pressure(hb + 1.0)
            assert p_lo == pytest.approx(p_hi, rel=1e-3)

    def test_density_scale_dispersion(self):
        """Monte Carlo density scaling must be exactly multiplicative."""
        a1 = USStandardAtmosphere1976(density_scale=1.0)
        a2 = USStandardAtmosphere1976(density_scale=1.15)
        for h in (0.0, 20e3, 50e3, 80e3):
            assert a2.density(h) == pytest.approx(1.15 * a1.density(h), rel=1e-9)

    def test_high_altitude_extrapolation_decays(self, atmo):
        assert atmo.density(120_000.0) < atmo.density(90_000.0) < atmo.density(84_000.0)

    def test_invalid_density_scale_rejected(self):
        with pytest.raises(ValueError):
            USStandardAtmosphere1976(density_scale=0.0)
