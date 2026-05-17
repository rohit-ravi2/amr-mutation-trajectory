"""Gate 0 — Environment Setup.

Verifies that all storage paths exist, the venv has all required packages,
and CUDA is available (or logs a warning and falls back to CPU). Appends a
single PASS/FAIL line to the gate log.

Idempotent: safe to re-run. Creates directories if missing; never overwrites.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stdout)
log = logging.getLogger("gate0")

SSD_ROOT = Path("/mnt/ssd/research/amr")
HDD_ROOT = Path("/mnt/hdd/research/amr")
GATE_LOG = SSD_ROOT / "logs" / "gates.log"

REQUIRED_DIRS = [
    SSD_ROOT / "src",
    SSD_ROOT / "processed",
    SSD_ROOT / "logs",
    SSD_ROOT / "results",
    SSD_ROOT / "models",
    SSD_ROOT / "embeddings",
    HDD_ROOT / "raw",
]

# (import_name, friendly_name)
REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("esm", "fair-esm"),
    ("transformers", "transformers"),
    ("Bio", "biopython"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("torch_geometric", "torch-geometric"),
    ("requests", "requests"),
    ("tqdm", "tqdm"),
    ("pyarrow", "pyarrow"),
]


def ensure_dirs() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        log.info("dir ok: %s", d)


def check_packages() -> list[str]:
    failures: list[str] = []
    for import_name, friendly in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "unknown")
            log.info("pkg ok: %s (%s)", friendly, ver)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{friendly}: {exc!r}")
            log.error("pkg FAIL: %s -> %s", friendly, exc)
    return failures


def check_cuda() -> tuple[bool, str]:
    import torch

    available = torch.cuda.is_available()
    if not available:
        log.warning("CUDA not available — CPU fallback in effect")
        return False, "cpu"
    name = torch.cuda.get_device_name(0)
    log.info("CUDA available: %s", name)
    # No-op tensor allocation, fp16, well under 7 GB free.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    _ = x.sum().item()
    del x
    torch.cuda.empty_cache()
    log.info("no-op fp16 tensor alloc/free succeeded")
    return True, name


def append_gate_log(line: str) -> None:
    GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with GATE_LOG.open("a") as f:
        f.write(f"{ts} {line}\n")


def main() -> int:
    log.info("Gate 0 starting; venv=%s", sys.prefix)
    ensure_dirs()
    pkg_failures = check_packages()
    if pkg_failures:
        append_gate_log("GATE_0: FAIL [packages_missing: " + "; ".join(pkg_failures) + "]")
        log.error("Gate 0 FAIL: missing packages: %s", pkg_failures)
        return 1
    cuda_ok, gpu_name = check_cuda()
    line = f"GATE_0: PASS [cuda: {cuda_ok}] [gpu: {gpu_name}] [venv: {sys.prefix}]"
    append_gate_log(line)
    log.info("Gate 0 PASS — wrote: %s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
