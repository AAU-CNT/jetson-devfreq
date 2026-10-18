"""Shared frequency/governor API for all clock controllers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from ._errors import DevfreqError, DevfreqFrequencyError, DevfreqGovernorError


@dataclass(order=True)
class FreqInfo:
    """A single frequency value."""
    freq_hz: int
    freq_mhz: float = field(init=False, compare=False)

    def __post_init__(self) -> None:
        self.freq_mhz = self.freq_hz / 1_000_000

    def __int__(self) -> int:
        return self.freq_hz

    def __repr__(self) -> str:
        return f"<FreqInfo {self.freq_mhz:.3f} MHz>"


#: A frequency in Hz, or a FreqInfo returned by list_frequencies().
Freq = Union[int, FreqInfo]


@dataclass(frozen=True)
class ClockState:
    """
    Saved governor and frequency bounds of one controller, from snapshot().

    detail holds backend-specific values (per-core CPU settings, EMC lock
    flags, ...). Everything is JSON-serialisable via as_dict()/from_dict(),
    so a state can be stored in run metadata and restored by another process.
    """
    controller: str
    governor: Optional[str]
    min_hz: int
    max_hz: int
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"controller": self.controller, "governor": self.governor,
                "min_hz": self.min_hz, "max_hz": self.max_hz, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClockState":
        return cls(controller=data["controller"], governor=data["governor"],
                   min_hz=int(data["min_hz"]), max_hz=int(data["max_hz"]),
                   detail=dict(data.get("detail", {})))


def _mhz(freq_hz: int) -> str:
    return f"{freq_hz / 1_000_000:g} MHz"


def _check_range(label: str, new_min: Optional[int], new_max: Optional[int],
                 cur_min: int, cur_max: int) -> None:
    """Reject a (partial) range update that would leave min above max."""
    eff_min = cur_min if new_min is None else new_min
    eff_max = cur_max if new_max is None else new_max
    if eff_min > eff_max:
        raise DevfreqFrequencyError(
            f"{label}: min {_mhz(eff_min)} would exceed max {_mhz(eff_max)}; "
            f"use set_freq_range() to move both bounds"
        )


def _ordered_range_write(new_min: Optional[int], new_max: Optional[int], cur_max: int,
                         write_min: Callable[[int], None],
                         write_max: Callable[[int], None]) -> None:
    """
    Write min/max in an order the kernel accepts: when the new floor is above
    the current ceiling, raise the ceiling first; otherwise lower the floor first.
    Callers must run _check_range() beforehand.
    """
    if new_min is not None and new_min > cur_max:
        if new_max is not None:
            write_max(new_max)
        write_min(new_min)
    else:
        if new_min is not None:
            write_min(new_min)
        if new_max is not None:
            write_max(new_max)


class ClockController(ABC):
    """
    Common frequency and governor API shared by the GPU, CPU and EMC controllers.

    Nothing here changes the governor implicitly: set_min_freq(), set_max_freq()
    and set_freq_range() only move the operating range, and the governor keeps
    scaling within it. pin_clock() is the one call that both selects a governor
    and collapses the range to a single frequency.

    All frequencies are in Hz; FreqInfo values are accepted wherever an int is.
    """

    #: Human-readable name used in error messages.
    name: str = "clock"
    #: Governors pin_clock() picks from (first available wins) when none is given.
    _PIN_GOVERNORS: tuple[str, ...] = ("userspace", "performance")
    #: Whether pin_clock() collapses the min/max range to the pinned frequency.
    _PIN_SETS_RANGE: bool = True

    # Backend hooks
    @abstractmethod
    def list_frequencies(self) -> list[FreqInfo]:
        """Return available frequencies sorted descending."""

    @abstractmethod
    def get_current_freq(self) -> FreqInfo:
        """Return the current frequency."""

    @abstractmethod
    def list_governors(self) -> list[str]:
        """Return available governors."""

    @abstractmethod
    def get_governor(self) -> Optional[str]:
        """Return the current governor, or None if the backend has none."""

    @abstractmethod
    def reset(self) -> None:
        """Restore the hardware frequency range and the default governor."""

    @abstractmethod
    def _write_governor(self, governor: str) -> None: ...

    @abstractmethod
    def _read_range(self) -> tuple[int, int]: ...

    @abstractmethod
    def _write_range(self, min_hz: Optional[int], max_hz: Optional[int]) -> None:
        """Write whichever bounds are not None, in a kernel-safe order."""

    def _after_pin(self, freq_hz: int, governor: Optional[str]) -> None:
        """Extra writes needed once the range is collapsed (e.g. userspace/set_freq)."""

    def _check_pinned(self, freq_hz: int) -> Optional[str]:
        """Return a description of why the clock is not at freq_hz, or None if it is."""
        return None

    def _snapshot_detail(self) -> dict[str, Any]:
        """Backend-specific state to save alongside governor and range."""
        return {}

    def _restore_detail(self, state: ClockState) -> None:
        """Restore what _snapshot_detail() saved; runs after range and governor."""

    @property
    def is_continuous(self) -> bool:
        """
        True when the backend has no discrete frequency table; list_frequencies()
        then returns only the bounds and any value within them is accepted.
        """
        return False

    # Public API
    def set_governor(self, governor: str) -> None:
        """Set the governor."""
        available = self.list_governors()
        if governor not in available:
            raise DevfreqGovernorError(
                f"{self.name} governor '{governor}' not available. Valid: {available}"
            )
        self._write_governor(governor)

    def get_min_freq(self) -> FreqInfo:
        """Return the current minimum frequency bound."""
        return FreqInfo(self._read_range()[0])

    def get_max_freq(self) -> FreqInfo:
        """Return the current maximum frequency bound."""
        return FreqInfo(self._read_range()[1])

    def set_min_freq(self, freq: Freq) -> None:
        """Set the minimum frequency bound. The governor is left unchanged."""
        freq_hz = int(freq)
        self._validate_freq(freq_hz)
        self._write_range(freq_hz, None)

    def set_max_freq(self, freq: Freq) -> None:
        """Set the maximum frequency bound. The governor is left unchanged."""
        freq_hz = int(freq)
        self._validate_freq(freq_hz)
        self._write_range(None, freq_hz)

    def set_freq_range(self, min_freq: Freq, max_freq: Freq) -> None:
        """
        Set both frequency bounds in one call. The governor is left unchanged
        and keeps scaling within [min_freq, max_freq].
        """
        min_hz, max_hz = int(min_freq), int(max_freq)
        if min_hz > max_hz:
            raise DevfreqFrequencyError(
                f"{self.name}: min {_mhz(min_hz)} is above max {_mhz(max_hz)}"
            )
        self._validate_freq(min_hz)
        self._validate_freq(max_hz)
        self._write_range(min_hz, max_hz)

    def pin_clock(self, freq: Freq, governor: Optional[str] = None,
                  settle_time: float = 0.5, verify: bool = True) -> None:
        """
        Pin the clock to exactly freq.

        Sets the governor (default: the first of "userspace"/"performance" that
        is available), sets min = max = freq (EMC instead writes the locked
        rate and leaves the range alone), and, with verify=True, waits
        settle_time seconds and checks that the clock actually settled.
        Undo with restore() of an earlier snapshot(), or reset() for hardware defaults.
        """
        freq_hz = int(freq)
        self._validate_freq(freq_hz)
        if governor is None:
            governor = self._default_pin_governor()
        if governor is not None:
            self.set_governor(governor)
        if self._PIN_SETS_RANGE:
            self._write_range(freq_hz, freq_hz)
        self._after_pin(freq_hz, governor)

        if verify:
            time.sleep(settle_time)
            problem = self._check_pinned(freq_hz)
            if problem:
                raise DevfreqFrequencyError(
                    f"{self.name} clock did not settle at {_mhz(freq_hz)}: {problem}"
                )

    def snapshot(self) -> ClockState:
        """
        Save the current governor and frequency bounds.

        Unlike reset(), which restores hardware defaults, restore(snapshot())
        returns the clock to exactly the state found, e.g. at the start of a
        benchmark run.
        """
        min_hz, max_hz = self._read_range()
        return ClockState(self.name, self.get_governor(), min_hz, max_hz, self._snapshot_detail())

    def restore(self, state: ClockState) -> None:
        """
        Restore a state saved by snapshot(). Only values that differ from the
        current ones are written.
        """
        self._check_state(state)
        if self._read_range() != (state.min_hz, state.max_hz):
            self._write_range(state.min_hz, state.max_hz)
        if state.governor is not None and state.governor != self.get_governor():
            self._write_governor(state.governor)
        self._restore_detail(state)

    # Internal helpers
    def _check_state(self, state: ClockState) -> None:
        if state.controller != self.name:
            raise DevfreqError(
                f"Cannot restore a {state.controller} snapshot on the {self.name} controller"
            )

    def _default_pin_governor(self) -> Optional[str]:
        available = self.list_governors()
        return next((g for g in self._PIN_GOVERNORS if g in available), None)

    def _validate_freq(self, freq_hz: int) -> None:
        freqs = [f.freq_hz for f in self.list_frequencies()]
        if not freqs:
            return
        if self.is_continuous:
            lo = min(freqs) if len(freqs) > 1 else 0
            hi = max(freqs)
            if not lo <= freq_hz <= hi:
                raise DevfreqFrequencyError(
                    f"{self.name} frequency {_mhz(freq_hz)} out of range "
                    f"[{_mhz(lo)}, {_mhz(hi)}]"
                )
        elif freq_hz not in freqs:
            raise DevfreqFrequencyError(
                f"{self.name} frequency {freq_hz} Hz not available. "
                f"Valid MHz values: {sorted(f / 1_000_000 for f in freqs)}"
            )
