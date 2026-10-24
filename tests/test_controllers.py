from __future__ import annotations

import json
from pathlib import Path

import pytest

from jetson_devfreq import (
    ClockState,
    Devfreq,
    DevfreqError,
    DevfreqFrequencyError,
    DevfreqGovernorError,
    DevfreqPermissionError,
    NvidiaSmiGpuController,
)

from conftest import EMC_DIR, GPU_DIR, GPU_FREQS, FakeNvidiaSmi, get, put

MHZ = 1_000_000
GPU = GPU_DIR.removeprefix("sys/")
EMC = EMC_DIR.removeprefix("sys/")


def cpu_node(n: int, name: str) -> str:
    return f"devices/system/cpu/cpu{n}/cpufreq/{name}"


# GPU (devfreq)

def test_gpu_range_leaves_governor(agx: Path, writes: list) -> None:
    gpu = Devfreq(str(agx)).gpu
    gpu.set_min_freq(510 * MHZ)
    gpu.set_max_freq(918 * MHZ)
    assert writes == [(f"{GPU}/min_freq", str(510 * MHZ)), (f"{GPU}/max_freq", str(918 * MHZ))]
    assert gpu.get_governor() == "nvhost_podgov"
    assert (gpu.get_min_freq().freq_hz, gpu.get_max_freq().freq_hz) == (510 * MHZ, 918 * MHZ)


def test_gpu_range_raise_writes_max_first(agx: Path, writes: list) -> None:
    gpu = Devfreq(str(agx)).gpu
    gpu.set_freq_range(306 * MHZ, 408 * MHZ)
    writes.clear()
    gpu.set_freq_range(918 * MHZ, 1224 * MHZ)
    assert [w[0].rsplit("/", 1)[1] for w in writes] == ["max_freq", "min_freq"]


def test_gpu_range_lower_writes_min_first(agx: Path, writes: list) -> None:
    gpu = Devfreq(str(agx)).gpu
    gpu.set_freq_range(918 * MHZ, 1224 * MHZ)
    writes.clear()
    gpu.set_freq_range(306 * MHZ, 408 * MHZ)
    assert [w[0].rsplit("/", 1)[1] for w in writes] == ["min_freq", "max_freq"]


def test_gpu_min_above_max_rejected(agx: Path, writes: list) -> None:
    gpu = Devfreq(str(agx)).gpu
    gpu.set_max_freq(408 * MHZ)
    writes.clear()
    with pytest.raises(DevfreqFrequencyError, match="set_freq_range"):
        gpu.set_min_freq(918 * MHZ)
    with pytest.raises(DevfreqFrequencyError):
        gpu.set_freq_range(918 * MHZ, 408 * MHZ)
    assert writes == []


def test_gpu_invalid_frequency(agx: Path) -> None:
    gpu = Devfreq(str(agx)).gpu
    with pytest.raises(DevfreqFrequencyError, match="not available"):
        gpu.pin_clock(300 * MHZ)
    with pytest.raises(DevfreqGovernorError):
        gpu.set_governor("bogus")


def test_gpu_pin_clock(agx: Path, writes: list) -> None:
    put(agx, f"{GPU_DIR}/cur_freq", 612 * MHZ)  # the fake kernel "settles" instantly
    gpu = Devfreq(str(agx)).gpu
    gpu.pin_clock(612 * MHZ, settle_time=0)
    assert writes == [
        (f"{GPU}/governor", "userspace"),
        (f"{GPU}/min_freq", str(612 * MHZ)),
        (f"{GPU}/max_freq", str(612 * MHZ)),
        (f"{GPU}/userspace/set_freq", str(612 * MHZ)),
    ]


def test_gpu_pin_clock_explicit_governor(agx: Path, writes: list) -> None:
    gpu = Devfreq(str(agx)).gpu
    gpu.pin_clock(gpu.list_frequencies()[0], governor="performance", verify=False)
    assert writes[0] == (f"{GPU}/governor", "performance")
    assert not any("set_freq" in path for path, _ in writes)


def test_gpu_pin_clock_verify_failure(agx: Path) -> None:
    gpu = Devfreq(str(agx)).gpu
    with pytest.raises(DevfreqFrequencyError, match="did not settle"):
        gpu.pin_clock(612 * MHZ, settle_time=0)


def test_gpu_reset(agx: Path) -> None:
    gpu = Devfreq(str(agx)).gpu
    gpu.pin_clock(612 * MHZ, verify=False)
    gpu.reset()
    assert gpu.get_governor() == "nvhost_podgov"
    assert (gpu.get_min_freq().freq_hz, gpu.get_max_freq().freq_hz) == (GPU_FREQS[0], GPU_FREQS[-1])


def test_permission_error_translated(agx: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = Devfreq(str(agx)).gpu

    def denied(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_text", denied)
    with pytest.raises(DevfreqPermissionError, match="sudo"):
        gpu.set_governor("performance")


# CPU (cpufreq)

def test_cpu_governor_all_cores(agx: Path) -> None:
    cpu = Devfreq(str(agx)).cpu
    cpu.set_governor("performance")
    for n in (0, 1, 2, 10):
        assert get(agx, f"sys/{cpu_node(n, 'scaling_governor')}") == "performance"


def test_cpu_range_in_khz_without_governor_change(agx: Path) -> None:
    cpu = Devfreq(str(agx)).cpu
    cpu.set_freq_range(729_600_000, 1_984_000_000)
    for n in (0, 1, 2, 10):
        assert get(agx, f"sys/{cpu_node(n, 'scaling_min_freq')}") == "729600"
        assert get(agx, f"sys/{cpu_node(n, 'scaling_max_freq')}") == "1984000"
    assert cpu.get_governor() == "schedutil"


def test_cpu_pin_and_reset(agx: Path) -> None:
    cpu = Devfreq(str(agx)).cpu
    cpu.pin_clock(1_497_600_000, settle_time=0)
    assert cpu.get_governor() == "userspace"
    assert cpu.get_min_freq() == cpu.get_max_freq()
    cpu.reset()
    assert cpu.get_governor() == "schedutil"
    assert (cpu.get_min_freq().freq_hz, cpu.get_max_freq().freq_hz) == (115_200_000, 2_201_600_000)


def test_cpu_hybrid_clamps_per_core(dgx: Path, no_nvidia_smi: None) -> None:
    cpu = Devfreq(str(dgx)).cpu
    assert cpu.is_continuous
    assert [f.freq_hz for f in cpu.list_frequencies()] == [3_900_000_000, 338_000_000]
    cpu.pin_clock(3_500_000_000, settle_time=0)
    assert get(dgx, f"sys/{cpu_node(0, 'scaling_max_freq')}") == "2808000"
    assert get(dgx, f"sys/{cpu_node(2, 'scaling_max_freq')}") == "3500000"
    with pytest.raises(DevfreqFrequencyError, match="out of range"):
        cpu.set_max_freq(4_000_000_000)


def test_cpu_range_check_is_all_or_nothing(dgx: Path, no_nvidia_smi: None, writes: list) -> None:
    cpu = Devfreq(str(dgx)).cpu
    # Lower P-core max only; E-cores keep 2.808 GHz
    put(dgx, f"sys/{cpu_node(2, 'scaling_max_freq')}", 1_000_000)
    with pytest.raises(DevfreqFrequencyError):
        cpu.set_min_freq(2_000_000_000)
    assert writes == []


# EMC (BPMP debugfs)

def test_emc_frequencies_khz_and_cap(agx: Path) -> None:
    put(agx, "sys/kernel/nvpmodel_clk_cap/emc", 2_133_000_000)
    emc = Devfreq(str(agx)).emc
    assert [f.freq_hz for f in emc.list_frequencies()] == [2_133_000_000, 665_600_000, 204_000_000]
    assert emc.get_cap().freq_hz == 2_133_000_000
    with pytest.raises(DevfreqFrequencyError):
        emc.pin_clock(3_199_000_000)


def test_emc_pin_and_reset(agx: Path, writes: list) -> None:
    emc = Devfreq(str(agx)).emc
    assert emc.get_governor() == "bwmgr"
    emc.pin_clock(665_600_000, settle_time=0)
    # Same sequence as jetson_clocks; min_rate/max_rate are not touched
    assert writes == [
        (f"{EMC}/clk/emc/mrq_rate_locked", "1"),
        (f"{EMC}/bwmgr/bwmgr_halt", "1"),
        (f"{EMC}/clk/emc/rate", "665600000"),
    ]
    assert emc.get_governor() == "locked"

    writes.clear()
    emc.reset()
    assert emc.get_governor() == "bwmgr"
    assert get(agx, f"{EMC_DIR}/bwmgr/bwmgr_halt") == "0"
    assert not any("min_rate" in p or "max_rate" in p for p, _ in writes)  # range already at hw bounds
    assert (emc.get_min_freq().freq_hz, emc.get_max_freq().freq_hz) == (204_000_000, 3_199_000_000)


@pytest.mark.parametrize("content", [
    "204000 665600 2133000 3199000",                     # no unit: small values taken as kHz
    "204000000 665600000 2133000000 3199000000",         # no unit: Hz
    "204 665.6 2133 3199 (MHz)",                         # non-integer tokens are skipped
])
def test_emc_possible_rates_formats(agx: Path, content: str) -> None:
    put(agx, f"{EMC_DIR}/emc/possible_rates", content)
    rates = [f.freq_hz for f in Devfreq(str(agx)).emc.list_frequencies()]
    expected = [3_199_000_000, 2_133_000_000, 665_600_000, 204_000_000]
    assert rates == ([3_199_000_000, 2_133_000_000, 204_000_000] if "665.6" in content else expected)


def test_emc_cap_survives_lock(agx: Path) -> None:
    # Locked, the cap node reads the locked rate; pinning back up must still work.
    emc = Devfreq(str(agx)).emc
    emc.pin_clock(204_000_000, settle_time=0)
    put(agx, "sys/kernel/nvpmodel_clk_cap/emc", 204_000_000)
    assert emc.get_cap().freq_hz == 3_199_000_000
    assert emc.list_frequencies()[0].freq_hz == 3_199_000_000
    emc.pin_clock(2_133_000_000, settle_time=0)
    assert emc.get_current_freq().freq_hz == 2_133_000_000


def test_emc_cap_unknown_when_found_locked(agx: Path) -> None:
    put(agx, f"{EMC_DIR}/clk/emc/mrq_rate_locked", 1)
    put(agx, "sys/kernel/nvpmodel_clk_cap/emc", 665_600_000)  # the locked rate, not the cap
    emc = Devfreq(str(agx)).emc
    assert emc.get_cap() is None
    assert len(emc.list_frequencies()) == 4
    emc.reset()  # unlocked: the cap node is trusted again
    put(agx, "sys/kernel/nvpmodel_clk_cap/emc", 2_133_000_000)
    assert emc.get_cap().freq_hz == 2_133_000_000


def test_emc_without_possible_rates(agx: Path) -> None:
    (agx / EMC_DIR / "emc/possible_rates").unlink()
    emc = Devfreq(str(agx)).emc
    assert emc.is_continuous
    assert [f.freq_hz for f in emc.list_frequencies()] == [3_199_000_000, 204_000_000]
    emc.set_max_freq(1_600_000_000)  # any value in range is accepted
    emc.reset()
    assert emc.get_max_freq().freq_hz == 3_199_000_000


# GPU (nvidia-smi)

def test_nvidia_smi_range_and_pin(dgx: Path, nvidia_smi: FakeNvidiaSmi) -> None:
    gpu = Devfreq(str(dgx)).gpu
    assert isinstance(gpu, NvidiaSmiGpuController)
    assert gpu.get_current_freq().freq_hz == 2418 * MHZ
    gpu.set_max_freq(1200 * MHZ)
    assert nvidia_smi.calls[-1] == ["nvidia-smi", "-i", "0", "-lgc", "0,1200"]
    gpu.pin_clock(1500 * MHZ, settle_time=0)
    assert nvidia_smi.calls[-1] == ["nvidia-smi", "-i", "0", "-lgc", "1500,1500"]
    assert gpu.get_min_freq().freq_hz == 1500 * MHZ
    with pytest.raises(DevfreqGovernorError):
        gpu.set_governor("performance")
    gpu.reset()
    assert nvidia_smi.calls[-1] == ["nvidia-smi", "-i", "0", "-rgc"]
    assert gpu.get_max_freq().freq_hz == 3003 * MHZ


def test_nvidia_smi_requires_root(dgx: Path, nvidia_smi: FakeNvidiaSmi) -> None:
    gpu = Devfreq(str(dgx)).gpu
    nvidia_smi.returncode, nvidia_smi.stderr = 4, "Insufficient Permissions. Are you root?"
    with pytest.raises(DevfreqPermissionError):
        gpu.pin_clock(1500 * MHZ, verify=False)


# snapshot() / restore()

def test_gpu_snapshot_restore(agx: Path) -> None:
    gpu = Devfreq(str(agx)).gpu
    gpu.set_freq_range(408 * MHZ, 1020 * MHZ)
    saved = gpu.snapshot()
    assert (saved.governor, saved.min_hz, saved.max_hz) == ("nvhost_podgov", 408 * MHZ, 1020 * MHZ)
    gpu.pin_clock(612 * MHZ, verify=False)
    gpu.restore(saved)
    assert gpu.snapshot() == saved


def test_gpu_restore_userspace_set_freq(agx: Path) -> None:
    gpu = Devfreq(str(agx)).gpu
    gpu.pin_clock(408 * MHZ, verify=False)
    saved = gpu.snapshot()
    assert saved.detail == {"set_freq_hz": 408 * MHZ}
    gpu.reset()
    put(agx, f"{GPU_DIR}/userspace/set_freq", "undefined")
    gpu.restore(saved)
    assert gpu.get_governor() == "userspace"
    assert get(agx, f"{GPU_DIR}/userspace/set_freq") == str(408 * MHZ)


def test_restore_unchanged_writes_nothing(agx: Path, writes: list) -> None:
    dfc = Devfreq(str(agx))
    dfc.restore_all(dfc.snapshot_all())
    assert writes == []


def test_restore_wrong_controller(agx: Path) -> None:
    dfc = Devfreq(str(agx))
    with pytest.raises(DevfreqError, match="GPU snapshot on the CPU"):
        dfc.cpu.restore(dfc.gpu.snapshot())


def test_cpu_snapshot_restore_per_core_json(agx: Path) -> None:
    cpu = Devfreq(str(agx)).cpu
    put(agx, f"sys/{cpu_node(10, 'scaling_governor')}", "performance")
    put(agx, f"sys/{cpu_node(10, 'scaling_min_freq')}", 1_984_000)
    saved = ClockState.from_dict(json.loads(json.dumps(cpu.snapshot().as_dict())))
    cpu.pin_clock(729_600_000, verify=False)
    cpu.restore(saved)
    assert get(agx, f"sys/{cpu_node(0, 'scaling_governor')}") == "schedutil"
    assert get(agx, f"sys/{cpu_node(10, 'scaling_governor')}") == "performance"
    assert get(agx, f"sys/{cpu_node(10, 'scaling_min_freq')}") == "1984000"
    assert get(agx, f"sys/{cpu_node(0, 'scaling_min_freq')}") == "115200"
    assert cpu.snapshot() == saved


def test_cpu_restore_core_mismatch(agx: Path) -> None:
    cpu = Devfreq(str(agx)).cpu
    saved = cpu.snapshot()
    del saved.detail["cores"]["cpu10"]
    with pytest.raises(DevfreqError, match="cores"):
        cpu.restore(saved)


def test_emc_snapshot_restore_locked(agx: Path, writes: list) -> None:
    # State found on the AGX: locked at the cap by jetson_clocks
    put(agx, f"{EMC_DIR}/clk/emc/mrq_rate_locked", 1)
    put(agx, f"{EMC_DIR}/bwmgr/bwmgr_halt", 1)
    put(agx, f"{EMC_DIR}/clk/emc/rate", 3_199_000_000)
    emc = Devfreq(str(agx)).emc
    saved = emc.snapshot()
    assert saved.governor == "locked"
    assert saved.detail == {"mrq_rate_locked": 1, "rate": 3_199_000_000, "bwmgr_halt": 1}

    emc.reset()
    put(agx, f"{EMC_DIR}/clk/emc/rate", 665_600_000)  # bwmgr scaled it down
    writes.clear()
    emc.restore(saved)
    assert writes == [
        (f"{EMC}/clk/emc/mrq_rate_locked", "1"),
        (f"{EMC}/bwmgr/bwmgr_halt", "1"),
        (f"{EMC}/clk/emc/rate", "3199000000"),
    ]


def test_emc_snapshot_restore_unlocked(agx: Path) -> None:
    emc = Devfreq(str(agx)).emc
    saved = emc.snapshot()
    emc.pin_clock(665_600_000, settle_time=0)
    emc.restore(saved)
    assert emc.get_governor() == "bwmgr"
    assert get(agx, f"{EMC_DIR}/bwmgr/bwmgr_halt") == "0"


def test_nvidia_smi_snapshot_restore(dgx: Path, nvidia_smi: FakeNvidiaSmi) -> None:
    gpu = Devfreq(str(dgx)).gpu
    unlocked = gpu.snapshot()
    gpu.pin_clock(1500 * MHZ, verify=False)
    locked = gpu.snapshot()
    gpu.restore(unlocked)
    assert nvidia_smi.calls[-1][-1] == "-rgc"
    gpu.restore(locked)
    assert nvidia_smi.calls[-1][-2:] == ["-lgc", "1500,1500"]


def test_restore_all_unknown_clock(dgx: Path, no_nvidia_smi: None, agx_states: dict) -> None:
    with pytest.raises(DevfreqError, match="emc"):
        Devfreq(str(dgx)).restore_all(agx_states)


@pytest.fixture
def agx_states(tmp_path_factory: pytest.TempPathFactory) -> dict:
    from conftest import make_agx
    return Devfreq(str(make_agx(tmp_path_factory.mktemp("agx")))).snapshot_all()
