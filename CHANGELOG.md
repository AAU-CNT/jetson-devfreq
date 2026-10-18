# Changelog

## 0.2.0 (unreleased)

First PyPI release. Renamed from `nvidia-devfreq` / `nvidia_devfreq`.

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
