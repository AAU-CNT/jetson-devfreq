"""Top-level Devfreq class."""

from __future__ import annotations

from typing import Optional, Union

from ._base import ClockController, ClockState
from ._errors import DevfreqError, DevfreqNotFoundError
from ._platform import JetsonPlatform
from .cpu import CpuFreqController
from .emc import EmcController
from .gpu import GpuFreqController, NvidiaSmiGpuController


class Devfreq:
    """
    Top-level class for NVIDIA platform frequency and governor control.

    Auto-detects the platform on construction and exposes one controller per
    clock domain. gpu is None when neither devfreq nor nvidia-smi is usable;
    emc is None off Jetson Orin, or when not running as root (see
    platform.emc_status). Requires root for any write operations.

    Example:
        dfc = Devfreq()
        dfc.gpu.set_freq_range(306_000_000, 918_000_000)
        dfc.cpu.pin_clock(1_984_000_000)
        dfc.reset_all()
    """

    def __init__(self, root: str = "/") -> None:
        self.platform: JetsonPlatform = JetsonPlatform.detect(root)
        self.gpu: Optional[Union[GpuFreqController, NvidiaSmiGpuController]] = self._build_gpu()
        self.cpu: CpuFreqController = CpuFreqController(self.platform.cpu_core_paths)
        self.emc: Optional[EmcController] = self._build_emc()

    @property
    def controllers(self) -> dict[str, ClockController]:
        """All available controllers, keyed by "gpu", "cpu" and "emc"."""
        found: dict[str, Optional[ClockController]] = {
            "gpu": self.gpu, "cpu": self.cpu, "emc": self.emc,
        }
        return {name: ctrl for name, ctrl in found.items() if ctrl is not None}

    def reset_all(self) -> None:
        """Restore every available clock to its hardware range and default governor."""
        for ctrl in self.controllers.values():
            ctrl.reset()

    def snapshot_all(self) -> dict[str, ClockState]:
        """Save the state of every available clock, keyed like controllers."""
        return {name: ctrl.snapshot() for name, ctrl in self.controllers.items()}

    def restore_all(self, states: dict[str, ClockState]) -> None:
        """
        Restore states saved by snapshot_all(). Clocks missing from states are
        left alone; a state for a clock this system lacks raises DevfreqError.
        """
        controllers = self.controllers
        missing = set(states) - set(controllers)
        if missing:
            raise DevfreqError(f"No controller for saved clock(s): {sorted(missing)}")
        for name, state in states.items():
            controllers[name].restore(state)

    def _build_gpu(self) -> Optional[Union[GpuFreqController, NvidiaSmiGpuController]]:
        if self.platform.gpu_devfreq_path:
            return GpuFreqController(self.platform.gpu_devfreq_path)
        # Fallback: try nvidia-smi (e.g. DGX Spark)
        try:
            return NvidiaSmiGpuController()
        except DevfreqNotFoundError:
            return None

    def _build_emc(self) -> Optional[EmcController]:
        if not self.platform.emc_debugfs_path:
            return None
        try:
            return EmcController(self.platform.emc_debugfs_path, self.platform.emc_cap_path)
        except DevfreqError:
            return None
