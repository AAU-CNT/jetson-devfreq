"""CPU clock controller (Linux cpufreq)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ._base import ClockController, ClockState, FreqInfo, _check_range, _mhz, _ordered_range_write
from ._errors import DevfreqError, DevfreqNotFoundError
from ._sysfs import _SysfsNode


class CpuFreqController(ClockController):
    """
    Controls CPU frequency and governor across all cores via cpufreq sysfs.

    All sysfs cpufreq files use kHz; this class converts to/from Hz on the
    public interface to keep it consistent with the GPU and EMC controllers.

    Reads are taken from the first core (all Jetson Orin cores share one
    frequency table). Writes are applied to every discovered core directory.

    Sysfs layout per core N (under /sys/devices/system/cpu/cpuN/cpufreq/):
        scaling_available_frequencies  — space-separated kHz values
        scaling_available_governors    — space-separated governor names
        scaling_governor               — current governor (writable)
        scaling_cur_freq               — current frequency in kHz (read-only)
        scaling_min_freq               — minimum clamp in kHz (writable)
        scaling_max_freq               — maximum clamp in kHz (writable)
        cpuinfo_max_freq               — hardware ceiling in kHz (read-only)
        cpuinfo_min_freq               — hardware floor in kHz (read-only)
    """

    name = "CPU"
    _DEFAULT_GOVERNORS: tuple[str, ...] = ("schedutil", "ondemand", "powersave")
    # ACPI CPPC has no userspace governor; powersave respects scaling_min/max clamps
    _PIN_GOVERNORS: tuple[str, ...] = ("userspace", "powersave", "performance")

    def __init__(self, core_paths: list[str]) -> None:
        if not core_paths:
            raise DevfreqNotFoundError("No CPU cpufreq sysfs paths found.")
        self._core_paths = core_paths
        # Reference core for reads (cpu0, or whichever is first)
        ref = core_paths[0]
        self._ref_avail_freqs  = _SysfsNode(f"{ref}/scaling_available_frequencies")
        self._ref_avail_govs   = _SysfsNode(f"{ref}/scaling_available_governors")
        self._ref_governor     = _SysfsNode(f"{ref}/scaling_governor")
        self._ref_cur_freq     = _SysfsNode(f"{ref}/scaling_cur_freq")
        self._ref_min_freq     = _SysfsNode(f"{ref}/scaling_min_freq")
        self._ref_max_freq     = _SysfsNode(f"{ref}/scaling_max_freq")
        self._ref_cpuinfo_min  = _SysfsNode(f"{ref}/cpuinfo_min_freq")
        # ACPI CPPC drivers (e.g. DGX Spark) provide no discrete frequency list
        self._has_discrete_freqs: bool = self._ref_avail_freqs.exists()

    @property
    def is_continuous(self) -> bool:
        return not self._has_discrete_freqs

    # Public API
    def list_governors(self) -> list[str]:
        """Return available CPU governors (read from the first core)."""
        return self._ref_avail_govs.read_list()

    def get_governor(self) -> Optional[str]:
        """Return the current CPU governor (read from the first core)."""
        return self._ref_governor.read() or None

    def list_frequencies(self) -> list[FreqInfo]:
        """Return available CPU frequencies in Hz, sorted descending.

        On ACPI CPPC platforms (e.g. DGX Spark) there is no discrete step list;
        returns [max, min] from cpuinfo as the frequency bounds.
        """
        if self._has_discrete_freqs:
            return sorted(
                [FreqInfo(self._khz_to_hz(int(k))) for k in self._ref_avail_freqs.read_list()],
                key=lambda f: f.freq_hz, reverse=True
            )
        # ACPI CPPC: no discrete steps — return hardware bounds.
        # Use the global max across all cores so hybrid platforms (e.g. DGX Spark
        # with P-cores at 3.9 GHz and E-cores at 2.8 GHz) report the correct ceiling.
        all_maxes = [
            self._khz_to_hz(_SysfsNode(f"{p}/cpuinfo_max_freq").read_int())
            for p in self._core_paths
            if _SysfsNode(f"{p}/cpuinfo_max_freq").exists()
        ]
        result = []
        if all_maxes:
            result.append(FreqInfo(max(all_maxes)))
        if self._ref_cpuinfo_min.exists():
            min_hz = self._khz_to_hz(self._ref_cpuinfo_min.read_int())
            if not result or min_hz != result[0].freq_hz:
                result.append(FreqInfo(min_hz))
        return result

    def get_current_freq(self) -> FreqInfo:
        """Return current CPU frequency (from the first core)."""
        return FreqInfo(self._khz_to_hz(self._ref_cur_freq.read_int()))

    def reset(self) -> None:
        """Restore all CPU cores to hardware bounds and default governor."""
        available_govs = self.list_governors()
        default_gov = next((g for g in self._DEFAULT_GOVERNORS if g in available_govs), None)

        for path in self._core_paths:
            hw_max_node = _SysfsNode(f"{path}/cpuinfo_max_freq")
            hw_min_node = _SysfsNode(f"{path}/cpuinfo_min_freq")
            if hw_max_node.exists() and hw_min_node.exists():
                min_node, max_node = self._core_range_nodes(path)
                _ordered_range_write(
                    self._khz_to_hz(hw_min_node.read_int()),
                    self._khz_to_hz(hw_max_node.read_int()),
                    self._khz_to_hz(max_node.read_int()),
                    self._khz_writer(min_node), self._khz_writer(max_node),
                )

            if default_gov:
                gov_node = _SysfsNode(f"{path}/scaling_governor")
                if gov_node.exists():
                    gov_node.write(default_gov)

    def restore(self, state: ClockState) -> None:
        """
        Restore a state saved by snapshot(), core by core. Only values that
        differ from the current ones are written.
        """
        self._check_state(state)
        saved = state.detail.get("cores", {})
        by_name = {Path(p).parent.name: p for p in self._core_paths}
        if set(saved) != set(by_name):
            raise DevfreqError(
                f"CPU snapshot covers cores {sorted(saved)}, this system has {sorted(by_name)}"
            )

        for name, core in saved.items():
            path = by_name[name]
            min_node, max_node = self._core_range_nodes(path)
            cur_max = self._khz_to_hz(max_node.read_int())
            current = (self._khz_to_hz(min_node.read_int()), cur_max)
            if current != (core["min_hz"], core["max_hz"]):
                _ordered_range_write(core["min_hz"], core["max_hz"], cur_max,
                                     self._khz_writer(min_node), self._khz_writer(max_node))
            gov_node = _SysfsNode(f"{path}/scaling_governor")
            if core["governor"] and gov_node.read() != core["governor"]:
                gov_node.write(core["governor"])

    # Backend hooks
    def _write_governor(self, governor: str) -> None:
        for path in self._core_paths:
            _SysfsNode(f"{path}/scaling_governor").write(governor)

    def _read_range(self) -> tuple[int, int]:
        return (self._khz_to_hz(self._ref_min_freq.read_int()),
                self._khz_to_hz(self._ref_max_freq.read_int()))

    def _write_range(self, min_hz: Optional[int], max_hz: Optional[int]) -> None:
        # Check every core before writing any, so a rejected range is never half-applied.
        plan = []
        for path in self._core_paths:
            min_node, max_node = self._core_range_nodes(path)
            cur_min = self._khz_to_hz(min_node.read_int())
            cur_max = self._khz_to_hz(max_node.read_int())
            lo, hi = self._clamp_to_core(path, min_hz), self._clamp_to_core(path, max_hz)
            _check_range(f"CPU ({Path(path).parent.name})", lo, hi, cur_min, cur_max)
            plan.append((lo, hi, cur_max, min_node, max_node))

        for lo, hi, cur_max, min_node, max_node in plan:
            _ordered_range_write(lo, hi, cur_max,
                                 self._khz_writer(min_node), self._khz_writer(max_node))

    def _check_pinned(self, freq_hz: int) -> Optional[str]:
        for path in self._core_paths:
            expected = self._clamp_to_core(path, freq_hz)
            min_node, max_node = self._core_range_nodes(path)
            lo = self._khz_to_hz(min_node.read_int())
            hi = self._khz_to_hz(max_node.read_int())
            if lo != expected or hi != expected:
                return (f"{Path(path).parent.name} scaling range is "
                        f"[{_mhz(lo)}, {_mhz(hi)}]")
        return None

    def _snapshot_detail(self) -> dict[str, Any]:
        cores = {}
        for path in self._core_paths:
            min_node, max_node = self._core_range_nodes(path)
            cores[Path(path).parent.name] = {
                "governor": _SysfsNode(f"{path}/scaling_governor").read() or None,
                "min_hz": self._khz_to_hz(min_node.read_int()),
                "max_hz": self._khz_to_hz(max_node.read_int()),
            }
        return {"cores": cores}

    # Internal helpers
    @staticmethod
    def _core_range_nodes(path: str) -> tuple[_SysfsNode, _SysfsNode]:
        return _SysfsNode(f"{path}/scaling_min_freq"), _SysfsNode(f"{path}/scaling_max_freq")

    def _clamp_to_core(self, path: str, freq_hz: Optional[int]) -> Optional[int]:
        """
        On hybrid platforms clamp each core to its own hardware ceiling so
        E-cores are not driven above their limit when targeting P-core freq.
        """
        if freq_hz is None or self._has_discrete_freqs:
            return freq_hz
        core_max_node = _SysfsNode(f"{path}/cpuinfo_max_freq")
        if not core_max_node.exists():
            return freq_hz
        return min(freq_hz, self._khz_to_hz(core_max_node.read_int()))

    @classmethod
    def _khz_writer(cls, node: _SysfsNode) -> Callable[[int], None]:
        return lambda freq_hz: node.write(cls._hz_to_khz(freq_hz))

    @staticmethod
    def _hz_to_khz(freq_hz: int) -> int:
        return freq_hz // 1000

    @staticmethod
    def _khz_to_hz(freq_khz: int) -> int:
        return freq_khz * 1000
