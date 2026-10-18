"""Diagnostic entry point: ``jetson-devfreq`` or ``python -m jetson_devfreq``."""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional

from . import ClockController, Devfreq, DevfreqError, NvidiaSmiGpuController, __version__


def _show(field: str, fn: Callable[[], object]) -> None:
    try:
        value = fn()
    except DevfreqError as exc:
        value = f"(error: {exc})"
    print(f"  {field:<12}: {value}")


def _frequencies(ctrl: ClockController) -> str:
    freqs = ctrl.list_frequencies()
    if not freqs:
        return "(none reported)"
    if ctrl.is_continuous:
        lo = freqs[-1].freq_mhz if len(freqs) > 1 else 0
        return f"{lo:g} – {freqs[0].freq_mhz:g} MHz (continuous)"
    return ", ".join(f"{f.freq_mhz:g}" for f in freqs) + " MHz"


def _report(label: str, ctrl: ClockController, backend: str) -> None:
    print(f"{label} ({backend})")
    _show("governor", lambda: ctrl.get_governor() or "(n/a)")
    _show("governors", lambda: ", ".join(ctrl.list_governors()) or "(none)")
    _show("frequencies", lambda: _frequencies(ctrl))
    _show("range", lambda: f"{ctrl.get_min_freq().freq_mhz:g} – {ctrl.get_max_freq().freq_mhz:g} MHz")
    _show("current", lambda: f"{ctrl.get_current_freq().freq_mhz:g} MHz")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jetson-devfreq",
        description="Show detected platform, governors and clock ranges (read-only).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--root", default="/", help=argparse.SUPPRESS)  # fake sysfs tree for tests
    args = parser.parse_args(argv)

    print(f"=== jetson-devfreq {__version__} diagnostic ===\n")
    try:
        dfc = Devfreq(root=args.root)
    except DevfreqError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    platform = dfc.platform
    print(f"Platform variant : {platform.variant.value}")
    print(f"Compatible       : {', '.join(platform.compatible_strings[:3]) or '(none)'}")
    print(f"GPU devfreq path : {platform.gpu_devfreq_path or '(not found)'}")
    print(f"CPU cores found  : {len(platform.cpu_core_paths)}")
    print(f"EMC (BPMP)       : {platform.emc_status}")
    print()

    if dfc.gpu is None:
        print("GPU: not available on this platform\n")
    elif isinstance(dfc.gpu, NvidiaSmiGpuController):
        _report("GPU", dfc.gpu, "nvidia-smi; any value up to max accepted")
    else:
        _report("GPU", dfc.gpu, "devfreq")
    _report("CPU", dfc.cpu, "cpufreq")
    if dfc.emc is not None:
        _report("EMC", dfc.emc, "BPMP debugfs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
