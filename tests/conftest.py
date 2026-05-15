# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# Trillium Technologies
# University of Oxford
# Giacomo Acciarini (giacomo.acciarini@gmail.com)
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.

"""Pytest configuration and fixtures."""

import os
import pytest
import numpy as np

import kessler


@pytest.fixture(scope="session")
def cdms_directory():
    """Get path to synthetic CDMs directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "docs", "notebooks", "synthetic_cdms")


@pytest.fixture(scope="session")
def cdm_dataset(cdms_directory):
    """Load CDM dataset."""
    return kessler.EventDataset(cdms_dir=cdms_directory, cdm_extension=".kvn")


@pytest.fixture(scope="session")
def cdms(cdm_dataset):
    """Get all CDMs from dataset."""
    return cdm_dataset.get_CDMs()


@pytest.fixture
def sample_cdm(cdms):
    """Get a single sample CDM."""
    if cdms:
        return cdms[0]
    return None


@pytest.fixture
def sample_state():
    """Create sample state vectors (position and velocity)."""
    # Position and velocity for target
    state_t = np.array([[0.0, 0.0, 0.0], [0.001, 0.002, 0.003]])

    # Position and velocity for chaser
    state_c = np.array([[0.05, 0.05, 0.0], [0.001, 0.002, 0.003]])

    return state_t, state_c


@pytest.fixture
def sample_covariance():
    """Create sample covariance matrices."""
    # 6x6 covariance matrix (position and velocity)
    # Position variances: 1000 m² to 10000 m²
    # Velocity variances: 0.01 to 0.1 (m/s)²
    cov = np.array(
        [
            [5000, 0, 0, 0, 0, 0],
            [0, 5000, 0, 0, 0, 0],
            [0, 0, 5000, 0, 0, 0],
            [0, 0, 0, 0.05, 0, 0],
            [0, 0, 0, 0, 0.05, 0],
            [0, 0, 0, 0, 0, 0.05],
        ]
    )
    return cov


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "cdm: marks tests that use CDM data")
    config.addinivalue_line("markers", "monte_carlo: marks tests for Monte Carlo methods")
    config.addinivalue_line("markers", "foster: marks tests for Foster collision probability")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test location/name
    for item in items:
        # Add unit marker to all tests by default
        if "unit" not in item.keywords and "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass

    yield
