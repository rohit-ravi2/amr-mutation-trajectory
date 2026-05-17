"""Extend Phase A5 — download additional ecoli + kpneumo genome FASTAs
to reach the 50 GB HDD target."""

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

RAW_DIR = Path("/mnt/hdd/research/amr/raw/patric_assemblies")
META_DIR = Path("/mnt/hdd/research/amr/raw")
LOG_PATH = Path("/mnt/ssd/research/amr/logs/day2/phaseA5_extend.log")

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

FTP_BASE = "ftp://ftp.bvbrc.org/genomes"

# Additional targets: extend ecoli to 25K and add kpneumo.
SPECIES_PLAN = [
    ("ecoli",   25000),  # currently 6000 done, will add ~19K more
    ("kpneumo", 8000),   # not started yet
]

N_WORKERS = 8


def setup_logger():
    log = logging.getLogger("a5x")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a"); fh.setFormatter(fmt); log.addHandler(fh)
    return log


def select_target_gids(species_key, n, log):
    meta = pd.read_parquet(META_DIR / f"genome_metadata_{species_key}.parquet")
    meta = meta.drop_duplicates("genome_id")
    meta = meta[meta["genome_length"].fillna(0) > 1_000_000].copy()
    meta = meta.sort_values("genome_length", ascending=False)
    return meta["genome_id"].astype(str).head(n).tolist()


def download_one(gid, out_path):
    if out_path.exists() and out_path.stat().st_size > 1024:
        return (gid, out_path.stat().st_size, True)
    url = f"{FTP_BASE}/{gid}/{gid}.fna"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    cmd = ["curl", "-ks", "--ftp-pasv", "--ssl-reqd", "--max-time", "120",
           "-o", str(tmp), url]
    try:
        rc = subprocess.run(cmd, timeout=130, capture_output=True).returncode
    except subprocess.TimeoutExpired:
        return (gid, 0, False)
    if rc != 0 or not tmp.exists() or tmp.stat().st_size < 1024:
        if tmp.exists(): tmp.unlink()
        return (gid, 0, False)
    with open(tmp, "rb") as fr, gzip.open(out_path, "wb", compresslevel=1) as fw:
        fw.write(fr.read())
    sz = out_path.stat().st_size
    tmp.unlink()
    return (gid, sz, True)


def main():
    log = setup_logger()
    log.info("=== Phase A5 extension ===")
    total_bytes = 0
    for sp_key, n_target in SPECIES_PLAN:
        out_dir = RAW_DIR / sp_key
        out_dir.mkdir(parents=True, exist_ok=True)
        gids = select_target_gids(sp_key, n_target, log)
        existing = {p.name.split(".fa.gz")[0] for p in out_dir.glob("*.fa.gz")}
        todo = [g for g in gids if g not in existing]
        log.info("[%s] target=%d existing=%d todo=%d", sp_key, n_target,
                 len(existing), len(todo))
        sp_t0 = time.time(); sp_ok = 0; sp_bytes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as exe:
            fmap = {exe.submit(download_one, gid, out_dir / f"{gid}.fa.gz"): gid
                    for gid in todo}
            with tqdm(total=len(todo), desc=f"{sp_key}-asm") as pbar:
                for fut in concurrent.futures.as_completed(fmap):
                    gid, sz, ok = fut.result()
                    pbar.update(1)
                    if ok:
                        sp_ok += 1; sp_bytes += sz
                    if pbar.n % 500 == 0:
                        elapsed = time.time() - sp_t0
                        log.info("[%s] %d/%d ok=%d  %.1f GB  (%.1f/s)",
                                 sp_key, pbar.n, len(todo), sp_ok,
                                 sp_bytes/1e9, pbar.n / (elapsed+1e-3))
        sp_total = sum(p.stat().st_size for p in out_dir.glob("*.fa.gz"))
        log.info("[%s] done %d files %.2f GB", sp_key,
                 len(list(out_dir.glob("*.fa.gz"))), sp_total / 1e9)
        total_bytes += sp_total
    log.info("Phase A5x total %.2f GB", total_bytes / 1e9)
    return 0


if __name__ == "__main__":
    sys.exit(main())
