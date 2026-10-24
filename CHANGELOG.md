# Changelog

## Unreleased

### Fixed
- `EmcController` failed on every ladder or pin call on L4T R36: BPMP `possible_rates` ends in a
  unit (`... 3199000 (kHz)`) that was parsed as a number. The unit is now read, and other
  non-integer tokens are skipped.
- After pinning EMC below the power-mode cap, higher rates were rejected as unavailable: while
  EMC is locked, `/sys/kernel/nvpmodel_clk_cap/emc` reads the locked rate. The cap is now only
  read while unlocked and remembered; `get_cap()` returns `None` if EMC has been locked since the
  controller was created, and the ladder is then not capped.

## 0.2.0 (2026-10-19)

First release.

### Added
- `EmcController` (`Devfreq().emc`): memory clock control on Jetson Orin via BPMP debugfs (root only).
- `set_min_freq()`, `set_max_freq()`, `set_freq_range()`, `get_min_freq()`, `get_max_freq()`,
  `get_governor()` on every controller. Range changes never touch the governor.
- `pin_clock(freq, governor=None)`: set governor and min = max in one call, with optional settle check.
- `snapshot()` / `restore()` on every controller and `Devfreq.snapshot_all()` / `restore_all()`:
  return clocks to exactly the state found, not just hardware defaults. `ClockState` is JSON-serialisable.
- `ClockController` base class, `is_continuous` property, `Devfreq.controllers`.
- `jetson-devfreq` console script (read-only diagnostic).

### Changed
- `Devfreq.gpu` is `None` instead of raising when neither devfreq nor nvidia-smi is available.
- nvidia-smi calls now honour `gpu_index`.
- Kernel-rejected sysfs writes raise `DevfreqError` instead of a bare `OSError`.

### Removed
- `set_frequency()` — use `pin_clock()` to pin, or `set_freq_range()` for an operating range.
- Unused `DeviceType` enum.
