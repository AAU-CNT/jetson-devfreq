"""Memory (EMC) clock controller for Jetson Orin via BPMP debugfs."""

from __future__ import annotations

from typing import Any, Optional

from ._base import ClockController, ClockState, FreqInfo, _check_range, _mhz, _ordered_range_write
from ._errors import DevfreqNotFoundError
from ._sysfs import _SysfsNode

# possible_rates values below this are taken to be kHz (EMC never runs under 10 MHz),
# when the file names no unit
_KHZ_THRESHOLD = 10_000_000
# Unit tokens possible_rates may end with, e.g. "204000 665600 2133000 3199000 (kHz)"
_UNITS = {"(hz)": 1, "(khz)": 1_000, "(mhz)": 1_000_000}


class EmcController(ClockController):
    """
    Controls the external memory controller (EMC/DRAM) clock on Jetson Orin.

    On L4T R36 the EMC clock is not a devfreq device; it is owned by the BPMP
    firmware and exposed through debugfs, the same interface jetson_clocks uses.
    debugfs is root-only, so every method (including reads) requires root.

    Debugfs layout (under /sys/kernel/debug/bpmp/debug/):
        clk/emc/rate              — current rate in Hz; target rate when locked (writable)
        clk/emc/min_rate          — minimum rate in Hz (writable)
        clk/emc/max_rate          — maximum rate in Hz (writable)
        clk/emc/mrq_rate_locked   — 1 = debugfs rate overrides kernel requests (writable)
        bwmgr/bwmgr_halt          — 1 = halt the BPMP bandwidth manager (writable, optional)
        emc/possible_rates        — supported rates, then a unit, e.g. "... 3199000 (kHz)" (optional)
    Power-mode cap: /sys/kernel/nvpmodel_clk_cap/emc (Hz, world-readable). While
    EMC is locked this node reads the locked rate instead (BPMP rounds every
    request to it), so the cap is only read while unlocked and remembered.

    EMC has no kernel governor. The two "governors" exposed here map onto the
    lock flags: "bwmgr" lets the bandwidth manager scale EMC with demand, and
    "locked" holds it at the rate written by pin_clock().
    """

    name = "EMC"
    GOVERNORS: tuple[str, ...] = ("bwmgr", "locked")
    _PIN_GOVERNORS: tuple[str, ...] = ("locked",)
    # Pin the jetson_clocks way (lock, halt bwmgr, write rate); leave min/max_rate alone
    _PIN_SETS_RANGE = False

    def __init__(self, debugfs_path: str, cap_path: Optional[str] = None) -> None:
        clk = f"{debugfs_path}/clk/emc"
        self._rate           = _SysfsNode(f"{clk}/rate")
        self._min_rate       = _SysfsNode(f"{clk}/min_rate")
        self._max_rate       = _SysfsNode(f"{clk}/max_rate")
        self._rate_locked    = _SysfsNode(f"{clk}/mrq_rate_locked")
        self._bwmgr_halt     = _SysfsNode(f"{debugfs_path}/bwmgr/bwmgr_halt")
        self._possible_rates = _SysfsNode(f"{debugfs_path}/emc/possible_rates")
        self._cap            = _SysfsNode(cap_path) if cap_path else None
        if not self._rate.exists():
            raise DevfreqNotFoundError(f"EMC clock not found under {clk}")
        # Bounds as found at construction; reset() falls back to these
        # when possible_rates is unavailable.
        self._initial_range = (self._min_rate.read_int(), self._max_rate.read_int())
        self._cap_hz: Optional[int] = None
        self.get_cap()  # remember the cap now, if EMC is unlocked

    @property
    def is_continuous(self) -> bool:
        return not self._possible_rates.exists()

    # Public API
    def get_cap(self) -> Optional[FreqInfo]:
        """
        Return the nvpmodel EMC cap for the current power mode, if exposed.

        The cap node is only trusted while EMC is unlocked; while locked, the
        last cap read unlocked by this controller is returned. None if EMC has
        been locked since this controller was created (the cap is unknown).
        """
        if self._cap is None or not self._cap.exists():
            return None
        if not self._rate_locked.read_int():
            self._cap_hz = self._cap.read_int()
        return FreqInfo(self._cap_hz) if self._cap_hz is not None else None

    def list_governors(self) -> list[str]:
        return list(self.GOVERNORS)

    def get_governor(self) -> Optional[str]:
        """Return "locked" if the rate is locked via debugfs, else "bwmgr"."""
        return "locked" if self._rate_locked.read_int() else "bwmgr"

    def list_frequencies(self) -> list[FreqInfo]:
        """
        Return supported EMC rates (at or below the power-mode cap), sorted descending.
        Without possible_rates, returns [max, min] bounds instead. Not capped when the
        cap is unknown (see get_cap()).
        """
        cap = self.get_cap()
        cap_hz = cap.freq_hz if cap else None
        if self._possible_rates.exists():
            rates = {r for r in self._hw_rates() if cap_hz is None or r <= cap_hz}
        else:
            lo, hi = self._initial_range
            rates = {lo, min(hi, cap_hz) if cap_hz else hi}
        return sorted((FreqInfo(r) for r in rates), key=lambda f: f.freq_hz, reverse=True)

    def get_current_freq(self) -> FreqInfo:
        """Return the current EMC rate."""
        return FreqInfo(self._rate.read_int())

    def reset(self) -> None:
        """Hand EMC back to the bandwidth manager and restore the hardware range."""
        self._write_governor("bwmgr")
        rates = self._hw_rates() if self._possible_rates.exists() else []
        lo, hi = (min(rates), max(rates)) if rates else self._initial_range
        if self._read_range() != (lo, hi):
            self._write_range(lo, hi)

    def restore(self, state: ClockState) -> None:
        """
        Restore a state saved by snapshot(): lock flags, locked rate and range.
        Only values that differ from the current ones are written.
        """
        self._check_state(state)
        if self._read_range() != (state.min_hz, state.max_hz):
            self._write_range(state.min_hz, state.max_hz)

        locked = state.detail.get("mrq_rate_locked", int(state.governor == "locked"))
        halt = state.detail.get("bwmgr_halt")
        if locked:
            # Lock before writing the rate so the write takes effect
            if self._rate_locked.read_int() != 1:
                self._rate_locked.write(1)
            self._restore_bwmgr_halt(halt)
            rate = state.detail.get("rate")
            if rate is not None and self._rate.read_int() != rate:
                self._rate.write(rate)
        else:
            self._restore_bwmgr_halt(halt)
            if self._rate_locked.read_int() != 0:
                self._rate_locked.write(0)

    # Backend hooks
    def _write_governor(self, governor: str) -> None:
        locked = int(governor == "locked")
        self._rate_locked.write(locked)
        if self._bwmgr_halt.exists():
            self._bwmgr_halt.write(locked)

    def _read_range(self) -> tuple[int, int]:
        return self._min_rate.read_int(), self._max_rate.read_int()

    def _write_range(self, min_hz: Optional[int], max_hz: Optional[int]) -> None:
        cur_min, cur_max = self._read_range()
        _check_range(self.name, min_hz, max_hz, cur_min, cur_max)
        _ordered_range_write(min_hz, max_hz, cur_max, self._min_rate.write, self._max_rate.write)

    def _after_pin(self, freq_hz: int, governor: Optional[str]) -> None:
        self._rate.write(freq_hz)

    def _check_pinned(self, freq_hz: int) -> Optional[str]:
        actual = self._rate.read_int()
        return None if actual == freq_hz else f"rate reads {_mhz(actual)}"

    def _snapshot_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "mrq_rate_locked": self._rate_locked.read_int(),
            "rate": self._rate.read_int(),
        }
        if self._bwmgr_halt.exists():
            detail["bwmgr_halt"] = self._bwmgr_halt.read_int()
        return detail

    # Internal helpers
    def _restore_bwmgr_halt(self, halt: Optional[int]) -> None:
        if halt is not None and self._bwmgr_halt.exists() and self._bwmgr_halt.read_int() != halt:
            self._bwmgr_halt.write(halt)

    def _hw_rates(self) -> list[int]:
        values, scale = [], None
        for token in self._possible_rates.read_list():
            if token.isdigit():
                values.append(int(token))
            elif token.lower() in _UNITS:
                scale = _UNITS[token.lower()]
        if scale is None:
            scale = 1000 if values and max(values) < _KHZ_THRESHOLD else 1
        return [v * scale for v in values]
