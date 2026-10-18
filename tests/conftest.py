"""Fake sysfs trees modelled on real AGX Orin (L4T R36) and DGX Spark layouts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jetson_devfreq import _sysfs

GPU_FREQS = [306_000_000, 408_000_000, 510_000_000, 612_000_000, 714_000_000, 816_000_000,
             918_000_000, 1_020_000_000, 1_122_000_000, 1_224_000_000, 1_300_500_000]
CPU_FREQS_KHZ = [115_200, 729_600, 1_497_600, 1_984_000, 2_201_600]
EMC_RATES_KHZ = [204_000, 665_600, 2_133_000, 3_199_000]

GPU_DIR = "sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu"
EMC_DIR = "sys/kernel/debug/bpmp/debug"


def put(root: Path, rel: str, value: object) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(f"{value}\n")


def get(root: Path, rel: str) -> str:
    return (root / rel).read_text().strip()


def make_cpus(root: Path, cores: dict[int, int], discrete: bool, governors: str) -> None:
    """cores maps core number to cpuinfo_max_freq (kHz)."""
    for n, max_khz in cores.items():
        d = f"sys/devices/system/cpu/cpu{n}/cpufreq"
        min_khz = CPU_FREQS_KHZ[0] if discrete else 338_000
        if discrete:
            put(root, f"{d}/scaling_available_frequencies", " ".join(map(str, CPU_FREQS_KHZ)))
        put(root, f"{d}/scaling_available_governors", governors)
        put(root, f"{d}/scaling_governor", "schedutil")
        put(root, f"{d}/scaling_cur_freq", max_khz)
        put(root, f"{d}/scaling_min_freq", min_khz)
        put(root, f"{d}/scaling_max_freq", max_khz)
        put(root, f"{d}/cpuinfo_min_freq", min_khz)
        put(root, f"{d}/cpuinfo_max_freq", max_khz)


def make_agx(root: Path) -> Path:
    put(root, "proc/device-tree/compatible",
        b"nvidia,p3737-0000+p3701-0005\x00nvidia,p3701-0005\x00nvidia,tegra234\x00")
    put(root, f"{GPU_DIR}/available_frequencies", " ".join(map(str, GPU_FREQS)))
    put(root, f"{GPU_DIR}/available_governors",
        "userspace tegra_wmark nvhost_podgov performance simple_ondemand")
    put(root, f"{GPU_DIR}/governor", "nvhost_podgov")
    put(root, f"{GPU_DIR}/cur_freq", GPU_FREQS[0])
    put(root, f"{GPU_DIR}/min_freq", GPU_FREQS[0])
    put(root, f"{GPU_DIR}/max_freq", GPU_FREQS[-1])
    put(root, f"{GPU_DIR}/userspace/set_freq", 0)
    # cpu10 checks numeric (not lexical) core ordering
    make_cpus(root, {0: 2_201_600, 1: 2_201_600, 2: 2_201_600, 10: 2_201_600}, discrete=True,
              governors="interactive conservative ondemand userspace powersave performance schedutil")
    put(root, f"{EMC_DIR}/clk/emc/rate", 2_133_000_000)
    put(root, f"{EMC_DIR}/clk/emc/min_rate", 204_000_000)
    put(root, f"{EMC_DIR}/clk/emc/max_rate", 3_199_000_000)
    put(root, f"{EMC_DIR}/clk/emc/mrq_rate_locked", 0)
    put(root, f"{EMC_DIR}/bwmgr/bwmgr_halt", 0)
    put(root, f"{EMC_DIR}/emc/possible_rates", " ".join(map(str, EMC_RATES_KHZ)))
    put(root, "sys/kernel/nvpmodel_clk_cap/emc", 3_199_000_000)
    return root


def make_dgx(root: Path) -> Path:
    put(root, "sys/class/dmi/id/product_name", "NVIDIA DGX Spark")
    # E-cores 0-1 at 2.808 GHz, P-cores 2-3 at 3.9 GHz
    make_cpus(root, {0: 2_808_000, 1: 2_808_000, 2: 3_900_000, 3: 3_900_000}, discrete=False,
              governors="conservative ondemand userspace powersave performance schedutil")
    return root


@pytest.fixture
def agx(tmp_path: Path) -> Path:
    return make_agx(tmp_path)


@pytest.fixture
def dgx(tmp_path: Path) -> Path:
    return make_dgx(tmp_path)


@pytest.fixture
def writes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record every sysfs write as (path relative to sysfs, value), in order."""
    log: list[tuple[str, str]] = []
    original = _sysfs._SysfsNode.write

    def recording_write(self: _sysfs._SysfsNode, value: object) -> None:
        original(self, value)
        rel = Path(self.path).as_posix().split("/sys/", 1)[-1]
        log.append((rel, str(value)))

    monkeypatch.setattr(_sysfs._SysfsNode, "write", recording_write)
    return log


@pytest.fixture
def no_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)


class FakeNvidiaSmi:
    """Stand-in for subprocess.run that answers nvidia-smi queries."""

    def __init__(self, max_mhz: int = 3003, cur_mhz: int = 2418, returncode: int = 0,
                 stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.answers = {"name": "NVIDIA GB10", "clocks.max.gr": str(max_mhz),
                        "clocks.gr": str(cur_mhz)}
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        query = next((a.split("=", 1)[1] for a in cmd if a.startswith("--query-gpu=")), None)
        if query is not None:
            return subprocess.CompletedProcess(cmd, 0, stdout=self.answers[query] + "\n", stderr="")
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="", stderr=self.stderr)


@pytest.fixture
def nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> FakeNvidiaSmi:
    fake = FakeNvidiaSmi()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake
