"""
jetson_devfreq — CPU, GPU and memory (EMC) clock and governor control for
NVIDIA Jetson Orin and DGX Spark.

Wraps Linux sysfs (devfreq for GPU, cpufreq for CPU), BPMP debugfs (EMC) and
nvidia-smi (DGX Spark GPU) behind one API. Requires root for write operations
and for anything EMC-related.

Usage:
```python
    dfc = Devfreq()
    print(dfc.platform.variant)
    print(dfc.gpu.list_frequencies())
    saved = dfc.snapshot_all()

    # Constrain the operating range; the governor keeps scaling within it
    dfc.gpu.set_freq_range(306_000_000, 918_000_000)

    # Or pin a clock: governor + min = max in one call
    dfc.gpu.pin_clock(612_000_000)
    dfc.cpu.set_governor("performance")

    # Put everything back exactly as found (or reset_all() for hardware defaults)
    dfc.restore_all(saved)
```
"""

from importlib.metadata import PackageNotFoundError, version as _version

from ._base import ClockController, ClockState, Freq, FreqInfo
from ._devfreq import Devfreq
from ._errors import (
    DevfreqError,
    DevfreqFrequencyError,
    DevfreqGovernorError,
    DevfreqNotFoundError,
    DevfreqPermissionError,
)
from ._platform import JetsonPlatform, JetsonVariant
from .cpu import CpuFreqController
from .emc import EmcController
from .gpu import GpuFreqController, NvidiaSmiGpuController

try:
    __version__ = _version("jetson-devfreq")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0+unknown"

__all__ = [
    "Devfreq",
    "DevfreqError",
    "DevfreqPermissionError",
    "DevfreqNotFoundError",
    "DevfreqGovernorError",
    "DevfreqFrequencyError",
    "ClockController",
    "ClockState",
    "Freq",
    "FreqInfo",
    "JetsonPlatform",
    "JetsonVariant",
    "GpuFreqController",
    "NvidiaSmiGpuController",
    "CpuFreqController",
    "EmcController",
    "__version__",
]
