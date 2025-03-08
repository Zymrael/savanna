import logging
import os

import pytest

from savanna.kernels.triton_src.cgcg.src.kernel_utils import DeviceProps


@pytest.fixture
def is_interpreter():
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


@pytest.fixture
def device_props():
    return DeviceProps()


@pytest.fixture(autouse=True)
def configure_logging():
    logging.basicConfig(level=logging.INFO)
