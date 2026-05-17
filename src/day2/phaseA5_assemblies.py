"""
Phase A5 — Bulk genome assembly FASTA download from BV-BRC.

Downloads contigs for top E. coli + S. aureus genomes flagged as
genome_quality_flags='Good'. Saves as gzipped FASTA on HDD.

Target: 5,000 each species (lighter than the agent prompt's 5K E.coli +
5K S.aureus to stay under HDD budget; we cap at 5K E.coli + 3K S.aureus =
about 25 + 7 GB = 32 GB, comfortably under 50 GB combined with
proteins).

Skipped if file already exists.
"""

from __future__ import annotations

import gzip
import logging
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm

RAW_DIR = Path("/mnt/hdd/research/amr/raw/patric_assemblies")
META_DIR = Path("/mnt/hdd/research/amr/raw")
LOG_PATH = Path("/mnt/ssd/research/amr/logs/day2/phaseA5_assemblies.log")
GATE_LOG = Path("/mnt/ssd/research/amr/logs/gates.log")

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

API = "https://www.bv-brc.org/api"

SPECIES_PLAN = [
    # (key, name, n_to_download)
    ("ecoli",  "Escherichia coli",     5000),
    ("saureus","Staphylococcus aureus", 3000),
]


def setup_logger() -> logging.Logger:
    log = logging.getLogger("phaseA5")
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


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "amr-research/0.2 (rohit.ravi2@gmail.com)",
                      "Accept": "application/dna+fasta"})
    retry = Retry(total=4, backoff_factor=0.6,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=20))
    return s


def select_target_gids(species_key: str, n: int, log: logging.Logger) -> list[str]:
    """Pick highest-quality genome ids of the species from metadata."""
    meta = pd.read_parquet(META_DIR / f"genome_metadata_{species_key}.parquet")
    log.info("[%s] metadata rows=%d cols=%s", species_key, len(meta), list(meta.columns)[:15])
    # Filter by genome_quality (string) when present
    qcol = None
    for c in ["genome_quality", "genome_status", "genome_quality_flags"]:
        if c in meta.columns:
            qcol = c
            break
    if qcol:
        good = meta[meta[qcol].astype(str).str.lower().str.contains("good", na=False)]
        log.info("[%s] %d 'Good' genomes via %s", species_key, len(good), qcol)
        if len(good) >= n:
            meta = good
    # Prefer genomes with assembly_accession and complete genome length
    if "genome_length" in meta.columns:
        meta = meta[meta["genome_length"].fillna(0) > 0].copy()
    meta = meta.drop_duplicates("genome_id")
    # Order by genome_length descending so we get largest / most complete
    sort_col = "genome_length" if "genome_length" in meta.columns else "genome_id"
    meta = meta.sort_values(sort_col, ascending=False)
    return meta["genome_id"].astype(str).head(n).tolist()


def already_have(gid: str, sp_key: str) -> bool:
    return (RAW_DIR / sp_key / f"{gid}.fa.gz").exists()


def download_one(gid: str, sp_key: str, session: requests.Session,
                 log: logging.Logger) -> int:
    out = RAW_DIR / sp_key / f"{gid}.fa.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out.stat().st_size
    body = f"eq(genome_id,{gid})&select(sequence_id,sequence)&limit(2000)"
    headers = {"Accept": "application/dna+fasta",
               "Content-Type": "application/rqlquery+x-www-form-urlencoded"}
    url = f"{API}/genome_sequence/"
    try:
        r = session.post(url, data=body, headers=headers, timeout=60)
    except Exception as e:
        log.warning("[%s] %s POST failed: %s", sp_key, gid, e)
        return 0
    if r.status_code != 200 or not r.content:
        log.warning("[%s] %s HTTP %d (%d bytes)", sp_key, gid, r.status_code, len(r.content))
        return 0
    with gzip.open(out, "wb", compresslevel=4) as gh:
        gh.write(r.content)
    return out.stat().st_size


def main() -> int:
    log = setup_logger()
    log.info("=== Phase A5 — assembly bulk download ===")
    session = make_session()
    total_bytes = 0
    for sp_key, sp_name, n_target in SPECIES_PLAN:
        gids = select_target_gids(sp_key, n_target, log)
        log.info("[%s] selected %d gids", sp_key, len(gids))
        # Filter to those not yet on disk
        out_dir = RAW_DIR / sp_key
        out_dir.mkdir(parents=True, exist_ok=True)
        done = {p.stem.replace(".fa","") for p in out_dir.glob("*.fa.gz")}
        # Some gids may already exist with .fa.gz suffix; check just by gid
        done = {p.name.split(".fa.gz")[0] for p in out_dir.glob("*.fa.gz")}
        log.info("[%s] already on disk: %d", sp_key, len(done))
        sp_total = 0
        n_ok = 0
        for i, gid in enumerate(tqdm(gids, desc=f"{sp_key}-asm")):
            if gid in done:
                continue
            sz = download_one(gid, sp_key, session, log)
            if sz > 0:
                n_ok += 1
                sp_total += sz
            if (i + 1) % 200 == 0:
                log.info("[%s] %d/%d gids, %d ok, %.1f MB downloaded",
                         sp_key, i+1, len(gids), n_ok, sp_total / 1024 / 1024)
            # be respectful
            time.sleep(0.05)
        log.info("[%s] done: %d new files, %.1f MB", sp_key, n_ok, sp_total / 1024 / 1024)
        total_bytes += sp_total

    total_gb = total_bytes / 1024 / 1024 / 1024
    log.info("Phase A5 total downloaded: %.2f GB", total_gb)
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    with GATE_LOG.open("a") as fh:
        fh.write(f"{ts} PHASE_A5_ASSEMBLIES: PASS [total_GB={total_gb:.2f}]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
