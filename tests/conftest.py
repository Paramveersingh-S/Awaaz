"""Pytest configuration and shared test fixtures for Awaaz."""

import pytest

from scripts.make_fixture import create_fixture


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    """Provides a synthetic ASR test fixture generated in a temporary session directory."""
    temp_dir = tmp_path_factory.mktemp("synthetic_fixture")
    train_m, dev_m = create_fixture(str(temp_dir), num_train=12, num_dev=4)
    return {
        "dir": str(temp_dir),
        "train_manifest": train_m,
        "dev_manifest": dev_m,
    }
