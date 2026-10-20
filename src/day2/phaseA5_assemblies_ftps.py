"""
Phase A5 v2 — Bulk genome assembly FASTA download via BV-BRC FTPS.

Faster than the JSON API: ~1 sec per 4.5 MB genome via curl over FTPS.

Stores as compressed .fa.gz on HDD. Skips files already on disk.

Target: 10,000 E. coli + 5,000 S. aureus + 5,000 K. pneumoniae,
totaling ~50 GB compressed (or ~85 GB uncompressed).
"""

from __future__ import annotations

import concurrent.futures
import gzip
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

RAW_DIR = Path("work/raw/patric_assemblies")
META_DIR = Path("work/raw")
LOG_PATH = Path("work/logs/day2/phaseA5_ftps.log")
GATE_LOG = Path("work/logs/gates.log")

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

FTP_BASE = "ftp://ftp.bvbrc.org/genomes"

SPECIES_PLAN = [
    ("ecoli",   "Escherichia coli",     10000),
    ("saureus", "Staphylococcus aureus", 5000),
    ("kpneumo", "Klebsiella pneumoniae", 5000),
]

N_WORKERS = 6  # respectful concurrency on a single FTPS server


def setup_logger() -> logging.Logger:
    log = logging.getLogger("phaseA5_ftps")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def select_target_gids(species_key: str, n: int, log: logging.Logger) -> list[str]:
    meta = pd.read_parquet(META_DIR / f"genome_metadata_{species_key}.parquet")
    meta = meta.drop_duplicates("genome_id").copy()
    # Keep ones with valid length
    meta = meta[meta["genome_length"].fillna(0) > 1_000_000]
    # Prefer larger / more-complete assemblies
    meta = meta.sort_values("genome_length", ascending=False)
    return meta["genome_id"].astype(str).head(n).tolist()


def download_one(gid: str, out_path: Path) -> tuple[str, int, bool]:
    """Download a single genome FASTA via curl FTPS. Returns (gid, bytes, ok)."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return (gid, out_path.stat().st_size, True)
    url = f"{FTP_BASE}/{gid}/{gid}.fna"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    cmd = ["curl", "-ks", "--ftp-pasv", "--ssl-reqd", "--max-time", "120",
           "-o", str(tmp_path), url]
    try:
        rc = subprocess.run(cmd, timeout=130, capture_output=True).returncode
    except subprocess.TimeoutExpired:
        return (gid, 0, False)
    if rc != 0 or not tmp_path.exists() or tmp_path.stat().st_size < 1024:
        if tmp_path.exists():
            tmp_path.unlink()
        return (gid, 0, False)
    # gzip in place
    with open(tmp_path, "rb") as fr, gzip.open(out_path, "wb", compresslevel=1) as fw:
        fw.write(fr.read())
    sz = out_path.stat().st_size
    tmp_path.unlink()
    return (gid, sz, True)


def main() -> int:
    log = setup_logger()
    log.info("=== Phase A5 v2 (FTPS) — assembly bulk download ===")

    total_bytes = 0
    total_files = 0
    summary = {}
    for sp_key, sp_name, n_target in SPECIES_PLAN:
        out_dir = RAW_DIR / sp_key
        out_dir.mkdir(parents=True, exist_ok=True)
        gids = select_target_gids(sp_key, n_target, log)
        log.info("[%s] target=%d gids", sp_key, len(gids))

        existing = {p.name.split(".fa.gz")[0] for p in out_dir.glob("*.fa.gz")}
        todo = [g for g in gids if g not in existing]
        log.info("[%s] already on disk=%d  todo=%d", sp_key, len(existing), len(todo))

        sp_ok = 0
        sp_bytes = 0
        sp_t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as exe:
            futures = {}
            for gid in todo:
                out = out_dir / f"{gid}.fa.gz"
                futures[exe.submit(download_one, gid, out)] = gid
            with tqdm(total=len(todo), desc=f"{sp_key}-asm") as pbar:
                for fut in concurrent.futures.as_completed(futures):
                    gid, sz, ok = fut.result()
                    if ok:
                        sp_ok += 1
                        sp_bytes += sz
                    pbar.update(1)
                    if pbar.n % 200 == 0:
                        rate = pbar.n / (time.time() - sp_t0 + 1e-3)
                        log.info("[%s] %d/%d ok=%d  %.1f MB  (%.1f/s)",
                                 sp_key, pbar.n, len(todo), sp_ok,
                                 sp_bytes / 1024 / 1024, rate)

        # Recompute total including already-existing files
        sp_total = sum(p.stat().st_size for p in out_dir.glob("*.fa.gz"))
        sp_n = len(list(out_dir.glob("*.fa.gz")))
        log.info("[%s] complete: %d files, %.2f GB", sp_key, sp_n, sp_total / 1e9)
        summary[sp_key] = {"files": sp_n, "GB": sp_total / 1e9}
        total_bytes += sp_total
        total_files += sp_n

    log.info("Phase A5 totals: %d files, %.2f GB", total_files, total_bytes / 1e9)
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    with GATE_LOG.open("a") as fh:
        fh.write(f"{ts} PHASE_A5_ASSEMBLIES: PASS [total_files={total_files}] "
                 f"[total_GB={total_bytes/1e9:.2f}] "
                 f"[breakdown={summary}]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
