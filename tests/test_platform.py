from __future__ import annotations

import os
from pathlib import Path

import pytest

from jetson_devfreq import Devfreq, EmcController, GpuFreqController, JetsonPlatform, JetsonVariant
from jetson_devfreq import _platform
from jetson_devfreq.__main__ import main

from conftest import put


def test_detect_agx(agx: Path) -> None:
    p = JetsonPlatform.detect(str(agx))
    assert p.variant is JetsonVariant.AGX_ORIN
    assert p.gpu_devfreq_path is not None and p.gpu_devfreq_path.endswith("17000000.gpu")
    assert [Path(c).parent.name for c in p.cpu_core_paths] == ["cpu0", "cpu1", "cpu2", "cpu10"]
    assert p.emc_status == "available"
    assert p.emc_cap_path is not None


@pytest.mark.parametrize("compatible, variant", [
    (b"nvidia,p3768-0000+p3767-0003\x00nvidia,p3767-0003\x00nvidia,tegra234\x00", JetsonVariant.ORIN_NANO),
    (b"nvidia,p3768-0000+p3767-0000\x00nvidia,p3767-0000\x00nvidia,tegra234\x00", JetsonVariant.ORIN_NX),
    (b"nvidia,tegra234\x00", JetsonVariant.TEGRA234),
])
def test_detect_orin_variants(tmp_path: Path, compatible: bytes, variant: JetsonVariant) -> None:
    put(tmp_path, "proc/device-tree/compatible", compatible)
    assert JetsonPlatform.detect(str(tmp_path)).variant is variant


def test_detect_dgx(dgx: Path) -> None:
    p = JetsonPlatform.detect(str(dgx))
    assert p.variant is JetsonVariant.DGX_SPARK
    assert p.gpu_devfreq_path is None
    assert p.emc_status == "unsupported platform"


def test_emc_requires_root(agx: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_stat = os.stat

    def stat(path: str, *args: object, **kwargs: object) -> os.stat_result:
        if "debug" in str(path):
            raise PermissionError(13, "Permission denied", path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(_platform.os, "stat", stat)
    dfc = Devfreq(str(agx))
    assert dfc.platform.emc_status == "requires root"
    assert dfc.emc is None


def test_devfreq_agx(agx: Path) -> None:
    dfc = Devfreq(str(agx))
    assert isinstance(dfc.gpu, GpuFreqController)
    assert isinstance(dfc.emc, EmcController)
    assert set(dfc.controllers) == {"gpu", "cpu", "emc"}


def test_devfreq_without_gpu(dgx: Path, no_nvidia_smi: None) -> None:
    dfc = Devfreq(str(dgx))
    assert dfc.gpu is None
    assert set(dfc.controllers) == {"cpu"}
    dfc.reset_all()


def test_diagnostic_main(agx: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(agx)]) == 0
    out = capsys.readouterr().out
    assert "agx_orin" in out
    assert "EMC (BPMP)       : available" in out
    assert "error" not in out
