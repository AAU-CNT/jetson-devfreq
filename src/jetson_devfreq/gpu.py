"""GPU clock controllers: Linux devfreq (Jetson) and nvidia-smi (DGX Spark)."""

from __future__ import annotations

import subprocess
from typing import Any, Optional

from ._base import ClockController, ClockState, FreqInfo, _check_range, _mhz, _ordered_range_write
from ._errors import DevfreqError, DevfreqGovernorError, DevfreqNotFoundError, DevfreqPermissionError
from ._sysfs import _SysfsNode


class GpuFreqController(ClockController):
    """
    Controls GPU frequency and governor via Linux devfreq sysfs.

    Sysfs layout (under <devfreq_root>/):
        available_frequencies   — space-separated list of supported Hz values
        available_governors     — space-separated list of governor names
        governor                — current governor (writable)
        cur_freq                — current frequency in Hz (read-only)
        min_freq                — minimum frequency clamp (writable)
        max_freq                — maximum frequency clamp (writable)
        userspace/set_freq      — target frequency when governor=userspace (writable)
    """

    name = "GPU"
    _DEFAULT_GOVERNORS: tuple[str, ...] = ("nvhost_podgov", "simple_ondemand")

    def __init__(self, devfreq_path: str) -> None:
        root = devfreq_path
        self._available_freqs  = _SysfsNode(f"{root}/available_frequencies")
        self._available_govs   = _SysfsNode(f"{root}/available_governors")
        self._governor         = _SysfsNode(f"{root}/governor")
        self._cur_freq         = _SysfsNode(f"{root}/cur_freq")
        self._min_freq         = _SysfsNode(f"{root}/min_freq")
        self._max_freq         = _SysfsNode(f"{root}/max_freq")
        self._set_freq         = _SysfsNode(f"{root}/userspace/set_freq")

    # Public API
    def list_governors(self) -> list[str]:
        """Return available GPU governors."""
        return self._available_govs.read_list()

    def get_governor(self) -> Optional[str]:
        """Return the current GPU governor."""
        return self._governor.read() or None

    def list_frequencies(self) -> list[FreqInfo]:
        """Return available GPU frequencies sorted descending."""
        return sorted(
            [FreqInfo(int(f)) for f in self._available_freqs.read_list()],
            key=lambda f: f.freq_hz, reverse=True
        )

    def get_current_freq(self) -> FreqInfo:
        """Return the current GPU frequency."""
        return FreqInfo(self._cur_freq.read_int())

    def reset(self) -> None:
        """Restore GPU to hardware bounds and default governor."""
        freqs = self.list_frequencies()
        if freqs:
            self._write_range(freqs[-1].freq_hz, freqs[0].freq_hz)

        available = self.list_governors()
        for default in self._DEFAULT_GOVERNORS:
            if default in available:
                self._governor.write(default)
                break

    # Backend hooks
    def _write_governor(self, governor: str) -> None:
        self._governor.write(governor)

    def _read_range(self) -> tuple[int, int]:
        return self._min_freq.read_int(), self._max_freq.read_int()

    def _write_range(self, min_hz: Optional[int], max_hz: Optional[int]) -> None:
        cur_min, cur_max = self._read_range()
        _check_range(self.name, min_hz, max_hz, cur_min, cur_max)
        _ordered_range_write(min_hz, max_hz, cur_max, self._min_freq.write, self._max_freq.write)

    def _after_pin(self, freq_hz: int, governor: Optional[str]) -> None:
        if governor == "userspace" and self._set_freq.exists():
            self._set_freq.write(freq_hz)

    def _check_pinned(self, freq_hz: int) -> Optional[str]:
        actual = self._cur_freq.read_int()
        return None if actual == freq_hz else f"cur_freq reads {_mhz(actual)}"

    def _snapshot_detail(self) -> dict[str, Any]:
        # userspace/ only exists while the userspace governor is active, and
        # reads "undefined" until a frequency has been set
        if self.get_governor() != "userspace":
            return {}
        set_freq = self._set_freq.read()
        return {"set_freq_hz": int(set_freq)} if set_freq.isdigit() else {}

    def _restore_detail(self, state: ClockState) -> None:
        set_freq_hz = state.detail.get("set_freq_hz")
        if state.governor == "userspace" and set_freq_hz is not None and self._set_freq.exists():
            self._set_freq.write(set_freq_hz)


# GPU frequency controller via nvidia-smi (fallback for platforms without devfreq)
class NvidiaSmiGpuController(ClockController):
    """
    Controls GPU frequency via nvidia-smi for platforms where Linux devfreq
    is unavailable (e.g. DGX Spark with GB10 Blackwell GPU).

    On some hardware (e.g. GB10) nvidia-smi -q -d SUPPORTED_CLOCKS reports N/A,
    but -lgc still works: the GPU accepts any MHz value and snaps to the nearest
    internal clock step. Use get_current_freq() under load to read the actual
    settled frequency.

    nvidia-smi has no governors and cannot read back locked clocks, so the
    range reported by get_min_freq()/get_max_freq() is the last one set through
    this object (0 to clocks.max.gr until then).

    Requires root for any range change and reset().
    """

    name = "GPU"
    _PIN_GOVERNORS: tuple[str, ...] = ()

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._locked: Optional[tuple[int, int]] = None
        try:
            self._query("name")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise DevfreqNotFoundError(
                "nvidia-smi not found or GPU not accessible"
            ) from exc

    @property
    def is_continuous(self) -> bool:
        return True

    def list_governors(self) -> list[str]:
        """nvidia-smi does not expose governors; returns empty list."""
        return []

    def get_governor(self) -> Optional[str]:
        """nvidia-smi does not expose governors; returns None."""
        return None

    def set_governor(self, governor: str) -> None:
        raise DevfreqGovernorError(
            "Governors are not supported via nvidia-smi; use set_freq_range() or pin_clock()."
        )

    def list_frequencies(self) -> list[FreqInfo]:
        """
        Return GPU frequency bounds as [max] from clocks.max.gr.

        On platforms where SUPPORTED_CLOCKS is N/A, only the hardware maximum
        is queryable. Any MHz value up to that maximum is accepted and the GPU
        will snap to the nearest internal step.
        """
        try:
            return [FreqInfo(int(self._query("clocks.max.gr")) * 1_000_000)]
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return []

    def get_current_freq(self) -> FreqInfo:
        """Return the current GPU graphics clock."""
        return FreqInfo(int(self._query("clocks.gr")) * 1_000_000)

    def reset(self) -> None:
        """Remove GPU clock lock and restore hardware-managed clocks. Requires root."""
        result = self._run("-rgc")
        if result.returncode != 0:
            self._raise_for(result, "-rgc")
        self._locked = None

    def restore(self, state: ClockState) -> None:
        """
        Restore a state saved by snapshot(): re-apply the clock lock, or remove
        it if none was set through this object. Requires root.
        """
        self._check_state(state)
        if state.detail.get("locked"):
            self._write_range(state.min_hz, state.max_hz)
        else:
            self.reset()

    # Backend hooks
    def _write_governor(self, governor: str) -> None:
        self.set_governor(governor)

    def _snapshot_detail(self) -> dict[str, Any]:
        return {"locked": self._locked is not None}

    def _read_range(self) -> tuple[int, int]:
        if self._locked is not None:
            return self._locked
        freqs = self.list_frequencies()
        return 0, freqs[0].freq_hz if freqs else 0

    def _write_range(self, min_hz: Optional[int], max_hz: Optional[int]) -> None:
        cur_min, cur_max = self._read_range()
        _check_range(self.name, min_hz, max_hz, cur_min, cur_max)
        lo = cur_min if min_hz is None else min_hz
        hi = cur_max if max_hz is None else max_hz
        result = self._run("-lgc", f"{lo // 1_000_000},{hi // 1_000_000}")
        if result.returncode != 0:
            self._raise_for(result, "-lgc")
        self._locked = (lo, hi)

    # nvidia-smi helpers
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["nvidia-smi", "-i", str(self._gpu_index), *args],
            capture_output=True, text=True,
        )

    def _query(self, field: str) -> str:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(self._gpu_index),
             f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()

    @staticmethod
    def _raise_for(result: subprocess.CompletedProcess[str], option: str) -> None:
        message = (result.stderr or result.stdout).strip()
        if "root" in message.lower() or "permission" in message.lower():
            raise DevfreqPermissionError(f"nvidia-smi {option} requires root (sudo)")
        raise DevfreqError(f"nvidia-smi {option} failed: {message}")
