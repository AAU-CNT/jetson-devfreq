# jetson-devfreq

Python control of CPU, GPU and memory (EMC) clocks and governors on NVIDIA Jetson Orin and DGX Spark. Use it for frequency sweeps, DVFS experiments and reproducible benchmarks.

It wraps the kernel interfaces directly and has no dependencies:

| Clock | Jetson Orin (AGX / NX / Nano) | DGX Spark |
|---|---|---|
| GPU | devfreq sysfs | `nvidia-smi -lgc` |
| CPU | cpufreq sysfs | cpufreq sysfs (ACPI CPPC, continuous) |
| EMC (memory) | BPMP debugfs, the same interface `jetson_clocks` uses | not supported |

> Not affiliated with or endorsed by NVIDIA.

## Install

Install the wheel from the [latest GitHub release](https://github.com/AAU-CNT/jetson-devfreq/releases/latest):

```sh
pip install https://github.com/AAU-CNT/jetson-devfreq/releases/download/v0.2.0/jetson_devfreq-0.2.0-py3-none-any.whl
# or straight from the tag:
pip install "git+https://github.com/AAU-CNT/jetson-devfreq@v0.2.0"
```

Changing clocks requires root, so run your script with `sudo`. If you use a virtualenv, run it as `sudo /path/to/venv/bin/python script.py`. EMC lives in debugfs, which is root-only even for reads. When running as a normal user, `Devfreq().emc` is `None`.

## Quick start

```python
from jetson_devfreq import Devfreq

dfc = Devfreq()
print(dfc.platform.variant)            # JetsonVariant.AGX_ORIN
print(dfc.gpu.list_frequencies())      # [<FreqInfo 1300.500 MHz>, ...]
saved = dfc.snapshot_all()             # governors, ranges, EMC lock state

# Operating range: the current governor keeps scaling, but only within the bounds
dfc.gpu.set_freq_range(306_000_000, 918_000_000)
dfc.cpu.set_max_freq(1_497_600_000)

# Pin: picks a governor and sets min = max in one call, then checks the clock settled
dfc.gpu.pin_clock(612_000_000)                    # userspace governor
dfc.cpu.pin_clock(1_984_000_000, governor="performance")
if dfc.emc:
    dfc.emc.pin_clock(dfc.emc.list_frequencies()[0])

# Governor only
dfc.cpu.set_governor("schedutil")

dfc.restore_all(saved)   # back to exactly what was found
# or: dfc.reset_all()    # hardware range + default governor on every clock
```

`snapshot()` returns a `ClockState`. Use `as_dict()` / `ClockState.from_dict()` to store it as JSON, for example in run metadata, and restore it later from another process.

The library never changes a governor unless you ask it to. The range setters leave the governor alone. Only `pin_clock()`, `restore()` and `reset()` change it.

## API

`dfc.gpu`, `dfc.cpu` and `dfc.emc` share one interface (`ClockController`). All frequencies are in Hz, and anywhere an `int` is accepted you can also pass a `FreqInfo` from `list_frequencies()`.

| Method | |
|---|---|
| `list_frequencies()` | Supported frequencies, highest first. On continuous backends (`is_continuous`), only the `[max, min]` bounds. |
| `get_current_freq()` | Current clock. |
| `list_governors()` / `get_governor()` / `set_governor(g)` | Governor control. |
| `get_min_freq()` / `get_max_freq()` | Current bounds. |
| `set_min_freq(f)` / `set_max_freq(f)` / `set_freq_range(lo, hi)` | Move the operating range. Writes go in a kernel-safe order. |
| `pin_clock(f, governor=None, settle_time=0.5, verify=True)` | Governor + min = max, then verify. |
| `snapshot()` / `restore(state)` | Save and restore the exact governor, range and backend state. Only changed values are written. |
| `reset()` | Hardware range + default governor. |

Errors are subclasses of `DevfreqError`: `DevfreqPermissionError`, `DevfreqNotFoundError`, `DevfreqGovernorError` and `DevfreqFrequencyError`.

### Backend notes

- **GPU (devfreq):** the default `pin_clock()` governor is `userspace`, which also writes `userspace/set_freq`. `reset()` restores `nvhost_podgov`.
- **CPU:** writes go to every core. On hybrid CPPC systems (DGX Spark), each core is clamped to its own `cpuinfo_max_freq`. The default `pin_clock()` governor is `userspace`, then `powersave`, then `performance`, whichever is available first.
- **EMC:** there is no kernel governor. The governors `"bwmgr"` (demand-scaled) and `"locked"` map onto BPMP's `mrq_rate_locked` / `bwmgr_halt`. `pin_clock()` uses the same sequence as `jetson_clocks` (lock, halt bwmgr, write `rate`) and leaves `min_rate`/`max_rate` alone. `reset()` hands control back to the bandwidth manager. Rates are capped by the current nvpmodel mode (`emc.get_cap()`).
- **GPU (nvidia-smi):** there are no governors. The range maps onto `-lgc min,max` and `reset()` runs `-rgc`. nvidia-smi can't read the lock back, so `get_min_freq()`/`get_max_freq()` and `snapshot()` only know about locks set through this object.

## Diagnostic

```sh
jetson-devfreq            # or: python -m jetson_devfreq
sudo jetson-devfreq       # includes EMC
```

The diagnostic only reads, so it is safe to run while a benchmark is in progress.

## Development

```sh
pip install -e .[test]
pytest
```

The tests run against a fake sysfs tree and don't need Jetson hardware.
