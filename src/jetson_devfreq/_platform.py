"""Platform detection and sysfs path discovery."""

from __future__ import annotations

import glob as _glob
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from ._sysfs import under_root


class JetsonVariant(str, Enum):
    AGX_ORIN  = "agx_orin"   # nvidia,p3701 — AGX Orin devkit / carrier
    ORIN_NX   = "orin_nx"    # nvidia,p3767 higher SKU — Orin NX
    ORIN_NANO = "orin_nano"  # nvidia,p3767 lower SKU — Orin Nano
    TEGRA234  = "tegra234"   # Generic: SoC confirmed, board unknown
    DGX_SPARK = "dgx_spark"  # NVIDIA DGX Spark (GB10, ACPI-based, no GPU devfreq)
    UNKNOWN   = "unknown"    # Not a recognised platform


_TEGRA234_VARIANTS = frozenset({
    JetsonVariant.AGX_ORIN, JetsonVariant.ORIN_NX, JetsonVariant.ORIN_NANO, JetsonVariant.TEGRA234,
})


@dataclass
class JetsonPlatform:
    """Auto-detected platform metadata."""
    variant: JetsonVariant
    compatible_strings: list[str]
    gpu_devfreq_path: Optional[str]  # None if GPU devfreq sysfs not found
    cpu_core_paths: list[str]        # sorted list of cpufreq sysfs dirs
    emc_debugfs_path: Optional[str] = None  # BPMP debugfs root, None if unusable
    emc_cap_path: Optional[str] = None      # nvpmodel EMC cap node, if present
    emc_status: str = "not found"           # "available", "requires root", "not found",
                                            # or "unsupported platform"
    root: str = "/"

    #Automatic variant and device detection (internal use or diagnostic)
    @classmethod
    def detect(cls, root: str = "/") -> "JetsonPlatform":
        """
        Auto-detect the platform by reading /proc/device-tree/compatible
        and discovering sysfs paths for GPU, CPU and EMC.

        root relocates every sysfs path (used for testing against a fake tree).
        """
        compatible_strings = cls._read_compatible(root)
        variant = cls._detect_variant(compatible_strings, root)
        emc_path, emc_cap, emc_status = cls._find_emc(variant, root)
        return cls(
            variant=variant,
            compatible_strings=compatible_strings,
            gpu_devfreq_path=cls._find_gpu_devfreq_path(root),
            cpu_core_paths=cls._find_cpu_core_paths(root),
            emc_debugfs_path=emc_path,
            emc_cap_path=emc_cap,
            emc_status=emc_status,
            root=root,
        )

    @staticmethod
    def _read_compatible(root: str = "/") -> list[str]:
        try:
            raw = Path(under_root(root, "/proc/device-tree/compatible")).read_bytes()
            return [s for s in raw.decode("ascii", errors="replace").split("\x00") if s]
        except OSError:
            return []

    @staticmethod
    def _detect_variant(compatible_strings: list[str], root: str = "/") -> JetsonVariant:
        if compatible_strings:
            joined = " ".join(compatible_strings)
            if "p3701" in joined:
                return JetsonVariant.AGX_ORIN
            if "p3767" in joined:
                # Distinguish NX (higher SKU) from Nano (lower SKU) by board ID suffix.
                # p3767-0000 / p3767-0001 → Orin NX; p3767-0003 / p3767-0004 → Orin Nano
                if any("p3767-000" in s and s.endswith(("3", "4")) for s in compatible_strings):
                    return JetsonVariant.ORIN_NANO
                return JetsonVariant.ORIN_NX
            if "tegra234" in joined:
                return JetsonVariant.TEGRA234
        # No device-tree compatible strings — try DMI (ACPI-based systems like DGX Spark)
        try:
            dmi = Path(under_root(root, "/sys/class/dmi/id/product_name")).read_text().strip().lower()
            if "dgx_spark" in dmi or "dgx spark" in dmi:
                return JetsonVariant.DGX_SPARK
        except OSError:
            pass
        return JetsonVariant.UNKNOWN

    @staticmethod
    def _find_gpu_devfreq_path(root: str = "/") -> Optional[str]:
        """
        Discover the GPU devfreq sysfs root without hardcoding an address.

        Strategy:
          1. Glob /sys/devices/platform/bus@0/*.gpu
          2. For each candidate, look for a devfreq/<name> subdirectory
          3. Fallback: use the candidate directory itself
          4. Verify available_frequencies is present
        """
        bus = _glob.escape(under_root(root, "/sys/devices/platform/bus@0"))
        for candidate in sorted(_glob.glob(os.path.join(bus, "*.gpu"))):
            devfreq_parent = Path(candidate) / "devfreq"
            entries = sorted(devfreq_parent.iterdir()) if devfreq_parent.is_dir() else []
            devfreq_root = str(entries[0]) if entries else candidate

            if (Path(devfreq_root) / "available_frequencies").exists():
                return devfreq_root

        return None

    @staticmethod
    def _find_cpu_core_paths(root: str = "/") -> list[str]:
        """
        Discover all CPU cpufreq sysfs directories, sorted by core number.
        Uses scaling_cur_freq as the presence check (works on both discrete
        cpufreq and continuous ACPI CPPC drivers that lack scaling_available_frequencies).
        """
        cpu_dir = _glob.escape(under_root(root, "/sys/devices/system/cpu"))
        raw = _glob.glob(os.path.join(cpu_dir, "cpu*", "cpufreq"))
        valid = [p for p in raw if (Path(p) / "scaling_cur_freq").exists()]

        def _core_number(p: str) -> int:
            # Extract the integer from e.g. /sys/.../cpu4/cpufreq
            name = Path(p).parent.name  # "cpu4"
            return int(name[3:]) if name[3:].isdigit() else 0

        return sorted(valid, key=_core_number)

    @staticmethod
    def _find_emc(variant: JetsonVariant, root: str = "/") -> tuple[Optional[str], Optional[str], str]:
        """
        Locate the BPMP debugfs EMC clock (Tegra234 only).
        Returns (debugfs_path, cap_path, status).
        """
        if variant not in _TEGRA234_VARIANTS:
            return None, None, "unsupported platform"
        cap = under_root(root, "/sys/kernel/nvpmodel_clk_cap/emc")
        cap_path = cap if os.path.exists(cap) else None
        debugfs = under_root(root, "/sys/kernel/debug/bpmp/debug")
        try:
            os.stat(os.path.join(debugfs, "clk", "emc", "rate"))
        except PermissionError:
            return None, cap_path, "requires root"
        except (FileNotFoundError, NotADirectoryError):
            return None, cap_path, "not found"
        return debugfs, cap_path, "available"
