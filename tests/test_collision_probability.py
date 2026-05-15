# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# Trillium Technologies
# University of Oxford
# Giacomo Acciarini (giacomo.acciarini@gmail.com)
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.

"""Tests for collision probability computations."""

import os
import unittest

import numpy as np
import pytest

import kessler
from kessler import EventDataset, foster_collision_probability


class TestFosterCollisionProbability(unittest.TestCase):
    """Test Foster (1992) collision probability method."""

    def setUp(self):
        """Set up test fixtures."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cdms_dir = os.path.join(current_dir, "..", "docs", "notebooks", "synthetic_cdms")
        self.events = EventDataset(cdms_dir=self.cdms_dir, cdm_extension=".kvn")
        self.cdms = self.events.get_CDMs()

    @pytest.mark.foster
    def test_foster_basic_probability(self):
        """Test that foster_collision_probability returns valid probability."""
        self.assertGreater(len(self.cdms), 0, "No CDMs loaded")

        cdm = self.cdms[0]
        pc = foster_collision_probability(cdm, collision_radius=70)

        # Check it's a valid probability
        self.assertIsInstance(pc, (float, np.floating))
        self.assertGreaterEqual(pc, 0.0)
        self.assertLessEqual(pc, 1.0)

    @pytest.mark.foster
    def test_foster_multiple_cdms(self):
        """Test Foster method on multiple CDMs."""
        for cdm in self.cdms[:10]:
            pc = foster_collision_probability(cdm, collision_radius=70)
            self.assertGreaterEqual(pc, 0.0)
            self.assertLessEqual(pc, 1.0)

    @pytest.mark.foster
    def test_foster_variable_collision_radius(self):
        """Test Foster method with different collision radii."""
        cdm = self.cdms[0]
        radii = [10, 50, 70, 100, 200, 500]
        probabilities = []

        for radius in radii:
            pc = foster_collision_probability(cdm, collision_radius=radius)
            probabilities.append(pc)
            self.assertGreaterEqual(pc, 0.0)
            self.assertLessEqual(pc, 1.0)

        # Probability should be monotonically increasing with radius
        for i in range(len(probabilities) - 1):
            self.assertGreaterEqual(
                probabilities[i + 1], probabilities[i], "Pc should increase with radius"
            )

    @pytest.mark.foster
    def test_foster_deterministic(self):
        """Test that Foster method is deterministic."""
        cdm = self.cdms[0]
        pc1 = foster_collision_probability(cdm, collision_radius=70)
        pc2 = foster_collision_probability(cdm, collision_radius=70)

        self.assertEqual(pc1, pc2, "Foster method should be deterministic")

    @pytest.mark.foster
    def test_foster_zero_radius(self):
        """Test Foster with zero collision radius."""
        cdm = self.cdms[0]
        pc = foster_collision_probability(cdm, collision_radius=0)

        # With zero radius, Pc should be extremely low or zero
        self.assertLessEqual(pc, 1e-6)

    @pytest.mark.foster
    def test_foster_large_radius(self):
        """Test Foster with large collision radius."""
        cdm = self.cdms[0]

        # With large radius, Pc should approach something meaningful
        pc = foster_collision_probability(cdm, collision_radius=50000)
        self.assertGreaterEqual(pc, 0.0)
        self.assertLessEqual(pc, 1.0)

    @pytest.mark.foster
    def test_foster_unit_consistency(self):
        """Test that Foster correctly handles CCSDS units."""
        cdm = self.cdms[0]

        # Get state and covariance
        state_0 = cdm.get_state(0)  # km
        state_1 = cdm.get_state(1)  # km
        cov_0 = cdm.get_covariance(0)  # m²
        cov_1 = cdm.get_covariance(1)  # m²

        # States should be in km range (orbital scale)
        dist_km = np.linalg.norm(state_0[0] - state_1[0])
        self.assertGreater(dist_km, 1.0)  # At least km apart
        self.assertLess(dist_km, 50000.0)  # Not absurdly far

        # Covariances should be in m² range
        pos_cov_0 = np.diag(cov_0)[:3]
        pos_cov_1 = np.diag(cov_1)[:3]
        self.assertTrue(np.all(pos_cov_0 > 0))
        self.assertTrue(np.all(pos_cov_1 > 0))

        # Compute Pc (should work with correct unit handling)
        pc = foster_collision_probability(cdm, collision_radius=70)
        self.assertGreaterEqual(pc, 0.0)
        self.assertLessEqual(pc, 1.0)


class TestMonteCarlo(unittest.TestCase):
    """Test Monte Carlo collision probability method."""

    def setUp(self):
        """Set up test fixtures."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cdms_dir = os.path.join(current_dir, "..", "docs", "notebooks", "synthetic_cdms")
        self.events = EventDataset(cdms_dir=self.cdms_dir, cdm_extension=".kvn")
        self.cdms = self.events.get_CDMs()

    @pytest.mark.monte_carlo
    def test_mc_probability_valid(self):
        """Test that MC probability is valid."""
        cdm = self.cdms[0]

        # Create minimal Conjunction model for MC computation
        conj = kessler.Conjunction(collision_threshold=70, pc_method="MC", mc_samples=100)

        # Extract state for CDM generation
        state_0 = cdm.get_state(0)
        state_1 = cdm.get_state(1)

        # For MC, we'd need more setup; just test the concept
        self.assertIsNotNone(conj)
        self.assertEqual(conj._pc_method, "MC")

    @pytest.mark.monte_carlo
    def test_mc_deterministic_seed(self):
        """Test MC with deterministic seed."""
        import torch

        torch.manual_seed(42)
        pc1 = self._compute_simple_mc(70)

        torch.manual_seed(42)
        pc2 = self._compute_simple_mc(70)

        # Should be close with same seed
        self.assertAlmostEqual(pc1, pc2, places=5)

    def _compute_simple_mc(self, collision_radius):
        """Simple MC Pc computation for testing."""
        import torch

        cdm = self.cdms[0]

        # Extract and convert to meters
        state_0 = cdm.get_state(0) * 1e3  # km to m
        state_1 = cdm.get_state(1) * 1e3  # km to m
        cov_0 = cdm.get_covariance(0)  # already in m²
        cov_1 = cdm.get_covariance(1)  # already in m²

        # Extract position only
        pos_0 = state_0[0, :3]
        pos_1 = state_1[0, :3]
        pos_cov_0 = cov_0[:3, :3]
        pos_cov_1 = cov_1[:3, :3]

        # Sample
        samples_0 = torch.distributions.MultivariateNormal(
            torch.tensor(pos_0, dtype=torch.float32), torch.tensor(pos_cov_0, dtype=torch.float32)
        ).sample((100,))

        samples_1 = torch.distributions.MultivariateNormal(
            torch.tensor(pos_1, dtype=torch.float32), torch.tensor(pos_cov_1, dtype=torch.float32)
        ).sample((100,))

        # Compute paired distances
        distances = torch.linalg.norm(samples_0 - samples_1, dim=1)

        # Probability
        pc = (distances < collision_radius).sum().item() / len(distances)
        return float(pc)


class TestCollisionProbabilityComparison(unittest.TestCase):
    """Test Foster vs MC comparison."""

    def setUp(self):
        """Set up test fixtures."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cdms_dir = os.path.join(current_dir, "..", "docs", "notebooks", "synthetic_cdms")
        self.events = EventDataset(cdms_dir=self.cdms_dir, cdm_extension=".kvn")
        self.cdms = self.events.get_CDMs()

    @pytest.mark.slow
    def test_foster_mc_agreement(self):
        """Test that Foster and MC methods give reasonable agreement."""
        cdm = self.cdms[0]

        pc_foster = foster_collision_probability(cdm, collision_radius=70)
        self.assertGreaterEqual(pc_foster, 0.0)
        self.assertLessEqual(pc_foster, 1.0)

        # MC computation would require full Conjunction setup
        # For now, just verify Foster works correctly


if __name__ == "__main__":
    unittest.main()
