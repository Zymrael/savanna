"""
check implementation of GlobalConfig for duplication errors (would overwrite)
"""
import pytest


@pytest.mark.cpu
def test_GlobalConfig_duplicates():
    """
    tests that there are no duplicates among parent classes of GlobalConfig
    """
    from savanna import GlobalConfig

    assert GlobalConfig.validate_keys(), "test_GlobalConfig_duplicates"
