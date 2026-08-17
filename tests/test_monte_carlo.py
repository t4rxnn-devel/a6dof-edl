"""Monte Carlo multi-factor dispersion tests."""

from __future__ import annotations

import numpy as np
import pytest

from a6dof_edl.simulation.monte_carlo import (
    DispersionModel,
    MonteCarloRunner,
    MonteCarloSummary,
)


class TestDispersionModel:
    def test_sample_keys(self):
        d = DispersionModel().sample(np.random.default_rng(0))
        assert set(d) == {"d_altitude", "d_velocity", "d_fpa_deg",
                          "density_scale", "ld_bias", "mass_scale"}

    def test_reproducibility(self):
        m = DispersionModel()
        a = m.sample(np.random.default_rng(7))
        b = m.sample(np.random.default_rng(7))
        assert a == b

    def test_statistical_scaling(self):
        m = DispersionModel(sigma_fpa_deg=0.2)
        rng = np.random.default_rng(1)
        draws = np.array([m.sample(rng)["d_fpa_deg"] for _ in range(4000)])
        assert draws.std() == pytest.approx(0.2, rel=0.1)
        assert abs(draws.mean()) < 0.02


@pytest.mark.integration
class TestCampaign:
    def test_small_campaign_sequential(self):
        mc = MonteCarloRunner(n_runs=2, seed=3, processes=1)
        s = mc.run()
        assert isinstance(s, MonteCarloSummary)
        assert s.n_runs == 2
        assert s.n_success == 2
        assert s.skip_suppression_rate == 1.0
        assert all(m.touchdown_vertical_speed_m_s <= 1.5 for m in s.per_run)
        assert s.peak_q_kPa_p99 <= 38.5

    def test_runner_validates_n_runs(self):
        with pytest.raises(ValueError):
            MonteCarloRunner(n_runs=0)
