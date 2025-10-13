import pytest
import unittest

from scipy.stats import norm
from tsetlin.utils.norm import norm_cdf

class TestNorm(unittest.TestCase):
    def test_norm(self):
        assert norm.cdf(0.0) == pytest.approx(norm_cdf(0.0), 0.000001)
        assert norm.cdf(0.1) == pytest.approx(norm_cdf(0.1), 0.000001)
        assert norm.cdf(0.15) == pytest.approx(norm_cdf(0.15), 0.000001)
        assert norm.cdf(0.3) == pytest.approx(norm_cdf(0.3), 0.000001)
        assert norm.cdf(0.35) == pytest.approx(norm_cdf(0.35), 0.000001)
        assert norm.cdf(0.5) == pytest.approx(norm_cdf(0.5), 0.000001)
        assert norm.cdf(0.55) == pytest.approx(norm_cdf(0.55), 0.000001)
        assert norm.cdf(0.7) == pytest.approx(norm_cdf(0.7), 0.000001)
        assert norm.cdf(0.75) == pytest.approx(norm_cdf(0.75), 0.000001)
        assert norm.cdf(0.9) == pytest.approx(norm_cdf(0.9), 0.000001)
        assert norm.cdf(0.95) == pytest.approx(norm_cdf(0.95), 0.000001)
        assert norm.cdf(1.00) == pytest.approx(norm_cdf(1.00), 0.000001)
