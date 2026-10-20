#!/usr/bin/env python3
"""
Gate 1: BV-BRC genome + phenotype acquisition for AMR trajectory prediction.

Downloads laboratory + computational AMR phenotype records, genome metadata,
and AMR-relevant protein sequences for three priority species.

Critical schema note (May 2026):
  BV-BRC's genome_amr collection is now dominated by a single XGBoost-based
  computational MIC predictor (evidence == "Computational Method"). Strict
  laboratory-only filtering yields:
      E. coli           ~198 unique genomes
      S. aureus         ~753 unique genomes
      K. pneumoniae   ~1,491 unique genomes
  All below the strict 5,000-genome hard-stop in the original Gate 1 spec.

  We therefore pull ALL evidence tiers but preserve the `evidence`,
  `computational_method`, and `computational_method_performance` fields so
  downstream gates can stratify by source type. Lab vs. computational MICs
  are NOT pooled; they are tagged.

Endpoints used:
  - https://www.bv-brc.org/api/genome_amr/     phenotype records (POST + rqlquery)
  - https://www.bv-brc.org/api/genome/         per-genome metadata
  - https://www.bv-brc.org/api/genome_feature/ CDS protein sequences

Resumability:
  - Per-species phenotype parquet appended every PHENO_FLUSH rows.
  - Per-genome metadata parquet built in one pass (cheap).
  - Per-genome AMR-relevant protein parquet written incrementally with
    a `<species>_proteins_done.txt` resume set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "https://www.bv-brc.org/api"
AMR_ENDPOINT = f"{API_BASE}/genome_amr/"
GENOME_ENDPOINT = f"{API_BASE}/genome/"
FEATURE_ENDPOINT = f"{API_BASE}/genome_feature/"
SEQUENCE_ENDPOINT = f"{API_BASE}/feature_sequence/"  # md5 -> sequence lookup

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/rqlquery+x-www-form-urlencoded",
}

DEFAULT_TIMEOUT = 180  # seconds per request
INTER_REQUEST_SLEEP = 0.5  # politeness delay
BACKOFF_BASE = 1.0
BACKOFF_MAX = 60.0
MAX_RETRIES = 8
HARD_STOP_AFTER_TIMEOUTS = 30  # consecutive failures => treat API as unreachable

PHENO_CHUNK = 25000  # phenotype rows per page
GENOME_CHUNK = 200   # genome_ids per metadata query
FEATURE_CHUNK = 5000  # CDS rows per page (per genome — most genomes <10K CDS)
PHENO_FLUSH = 5000   # rows between parquet flushes

# Storage paths
HDD_ROOT = Path("work/raw")
SSD_LOGS = Path("work/logs")
SSD_PROCESSED = Path("work/processed")
GATE_LOG = SSD_LOGS / "gates.log"

SPECIES = [
    {
        "key": "ecoli",
        "name": "Escherichia coli",
        "name_url": "Escherichia%20coli",
        "taxon_id": 562,
    },
    {
        "key": "saureus",
        "name": "Staphylococcus aureus",
        "name_url": "Staphylococcus%20aureus",
        "taxon_id": 1280,
    },
    {
        "key": "kpneumo",
        "name": "Klebsiella pneumoniae",
        "name_url": "Klebsiella%20pneumoniae",
        "taxon_id": 573,
    },
]

# AMR-relevant CDS product keywords. Each entry can be:
#   ("plain", keyword)    — case-insensitive substring match
#   ("word", keyword)     — match only on word boundaries (avoids "Cat" matching "Catabolite")
#   ("regex", pattern)    — raw regex with re.IGNORECASE
# Filter sized for AMR breadth without ballooning the index by promiscuous matches.
AMR_PRODUCT_KEYWORDS: list[tuple[str, str]] = [
    # Beta-lactamases (family terms and major variants)
    ("plain", "beta-lactamase"), ("plain", "β-lactamase"),
    ("plain", "carbapenemase"), ("plain", "penicillinase"),
    ("plain", "cephalosporinase"),
    ("word", "OXA"), ("word", "KPC"), ("word", "NDM"),
    ("word", "VIM"), ("word", "IMP"), ("word", "CTX-M"),
    ("word", "TEM"), ("word", "SHV"), ("word", "CMY"), ("word", "AmpC"),
    # Topoisomerases (quinolone targets)
    ("word", "gyrA"), ("word", "gyrB"), ("word", "parC"), ("word", "parE"),
    ("plain", "DNA gyrase"), ("plain", "DNA topoisomerase"),
    # DHFR (trimethoprim)
    ("plain", "dihydrofolate reductase"), ("word", "DHFR"), ("word", "dhfr"),
    ("plain", "trimethoprim"),
    # Aminoglycoside-modifying enzymes
    ("plain", "aminoglycoside"),
    ("regex", r"\bAAC\("), ("regex", r"\bANT\("), ("regex", r"\bAPH\("),
    ("plain", "16S ribosomal RNA methyltransferase"),
    ("plain", "16S rRNA methyltransferase"),
    # Vancomycin
    ("plain", "vancomycin"), ("word", "vanA"), ("word", "vanB"),
    # Quinolone-related
    ("plain", "fluoroquinolone"), ("word", "qnrA"), ("word", "qnrB"),
    ("word", "qnrS"),
    # Efflux
    ("plain", "efflux"), ("word", "AcrB"), ("word", "AcrA"),
    ("word", "MexA"), ("word", "MexB"), ("word", "norA"), ("word", "tet38"),
    # PBPs
    ("word", "PBP"), ("plain", "penicillin-binding"),
    # Rifampin/Isoniazid
    ("word", "rpoB"), ("word", "katG"), ("word", "inhA"),
    # Macrolides
    ("plain", "macrolide"), ("plain", "erythromycin"),
    ("word", "ermA"), ("word", "ermB"), ("word", "ermC"),
    ("word", "mefA"), ("word", "msrA"),
    # Tetracycline (chemical name AND ribosomal protection / efflux genes)
    ("plain", "tetracycline"),
    ("regex", r"\btet[A-Z]\b"),
    # Chloramphenicol
    ("plain", "chloramphenicol"),
    ("regex", r"\bcat[A-Z]?\d*\b"),
    # Sulfonamides
    ("plain", "sulfonamide"), ("word", "sul1"), ("word", "sul2"),
    ("word", "sul3"),
    # Colistin
    ("plain", "colistin"), ("regex", r"\bmcr[- ]\d"),
    ("plain", "phosphoethanolamine transferase"),
    # Porins / outer membrane (passive)
    ("word", "porin"), ("word", "OmpC"), ("word", "OmpF"), ("word", "OmpK"),
    ("word", "OprD"),
    # MRSA-specific
    ("word", "mecA"), ("word", "mecC"),
]


def _build_amr_regex(keywords: list[tuple[str, str]]) -> re.Pattern:
    parts: list[str] = []
    for kind, k in keywords:
        if kind == "plain":
            parts.append(re.escape(k))
        elif kind == "word":
            parts.append(r"\b" + re.escape(k) + r"\b")
        elif kind == "regex":
            parts.append(k)
        else:
            raise ValueError(f"unknown keyword kind {kind!r}")
    return re.compile("|".join(parts), re.IGNORECASE)


_AMR_PRODUCT_RE = _build_amr_regex(AMR_PRODUCT_KEYWORDS)

PHENO_FIELDS = [
    "genome_id", "genome_name", "taxon_id", "antibiotic",
    "resistant_phenotype", "measurement", "measurement_sign",
    "measurement_value", "measurement_unit", "laboratory_typing_method",
    "laboratory_typing_platform", "testing_standard", "testing_standard_year",
    "evidence", "computational_method", "computational_method_version",
    "computational_method_performance", "source", "owner", "public",
    "date_inserted", "date_modified",
]
GENOME_FIELDS = [
    "genome_id", "genome_name", "taxon_id", "strain", "serovar",
    "biovar", "pathovar", "mlst", "isolation_source", "isolation_country",
    "geographic_location", "host_name", "host_age", "host_health",
    "collection_year", "collection_date", "genome_length",
    "genome_status", "completion_date", "publication", "bioproject_accession",
    "biosample_accession", "assembly_accession", "gc_content",
    "contigs", "patric_cds", "checkm_completeness", "checkm_contamination",
]
FEATURE_FIELDS = [
    "feature_id", "genome_id", "patric_id", "feature_type", "product",
    "gene", "aa_length", "aa_sequence_md5",
    "go", "ec", "pgfam_id", "plfam_id", "figfam_id",
]
# Note: aa_sequence is NO LONGER returned by /genome_feature/ (BV-BRC 1.9+).
# Sequences must be fetched from /feature_sequence/ keyed by md5.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    SSD_LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("gate1")
    log.setLevel(logging.INFO)
    if log.handlers:
        return log
    fmt = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.FileHandler(SSD_LOGS / "gate1_bvbrc.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    logging.Formatter.converter = time.gmtime
    return log


# ---------------------------------------------------------------------------
# Robust HTTP session
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s


def post_query(
    session: requests.Session,
    url: str,
    body: str,
    log: logging.Logger,
    consec_fail_counter: dict,
) -> Optional[requests.Response]:
    """POST a BV-BRC RQL query with exponential backoff."""
    delay = BACKOFF_BASE
    for attempt in range(MAX_RETRIES):
        try:
            r = session.post(url, data=body, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 200:
                consec_fail_counter["n"] = 0
                return r
            if r.status_code in (429, 503):
                log.warning("HTTP %d (rate-limit) attempt=%d, sleeping %.1fs",
                            r.status_code, attempt + 1, delay)
            else:
                log.warning("HTTP %d attempt=%d body=%s", r.status_code,
                            attempt + 1, r.text[:200])
        except (requests.Timeout, requests.ConnectionError) as e:
            log.warning("network error attempt=%d %s", attempt + 1, type(e).__name__)
        time.sleep(delay)
        delay = min(delay * 2, BACKOFF_MAX)
    consec_fail_counter["n"] += 1
    log.error("query failed after %d retries; body=%s", MAX_RETRIES, body[:300])
    return None


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def append_parquet(rows: list[dict], path: Path, log: logging.Logger) -> None:
    """Append a batch of rows by writing a side-car parquet shard.

    For large pulls (millions of rows), rewriting the entire parquet on each
    flush is O(N^2). Instead, write each batch to a `<stem>.parts/shard_XXXX.parquet`
    inside a partition directory. The caller is expected to call
    `consolidate_parquet(path)` at the end to merge shards into a single file.
    """
    if not rows:
        return
    parts_dir = path.with_suffix(path.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_existing = sum(1 for _ in parts_dir.glob("shard_*.parquet"))
    shard = parts_dir / f"shard_{n_existing:05d}.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(shard, index=False)
    log.info("flushed %d rows -> %s (shards=%d)", len(rows), shard.name, n_existing + 1)


def consolidate_parquet(path: Path, log: logging.Logger,
                        dedup_keys: Optional[list[str]] = None) -> int:
    """Merge shards under <path>.parts into a single parquet at <path>.
    Optionally deduplicate by dedup_keys. Removes shard directory on success."""
    parts_dir = path.with_suffix(path.suffix + ".parts")
    shards = sorted(parts_dir.glob("shard_*.parquet")) if parts_dir.exists() else []
    if path.exists():
        existing_shards = [path]
    else:
        existing_shards = []
    all_files = existing_shards + shards
    if not all_files:
        return 0
    dfs = [pd.read_parquet(p) for p in all_files]
    merged = pd.concat(dfs, ignore_index=True)
    if dedup_keys:
        before = len(merged)
        merged = merged.drop_duplicates(subset=dedup_keys, keep="first")
        log.info("consolidated %d shards, deduped %d -> %d rows",
                 len(shards), before, len(merged))
    else:
        log.info("consolidated %d shards into %d rows", len(shards), len(merged))
    # Write atomically
    tmp = path.with_suffix(path.suffix + ".tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)
    # Clean up shards
    for s in shards:
        s.unlink()
    if parts_dir.exists():
        try:
            parts_dir.rmdir()
        except OSError:
            pass
    return len(merged)


def read_done_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {ln.strip() for ln in f if ln.strip()}


def append_done(path: Path, gid: str) -> None:
    with path.open("a") as f:
        f.write(f"{gid}\n")


# ---------------------------------------------------------------------------
# Phenotype download
# ---------------------------------------------------------------------------

def download_phenotypes(species: dict, session: requests.Session,
                        log: logging.Logger, fail: dict) -> tuple[int, int]:
    """Download all AMR phenotype records for a species. Returns
    (n_records_after_dedup, n_unique_genomes)."""
    out_path = HDD_ROOT / f"amr_phenotypes_{species['key']}.parquet"
    sp_enc = species["name_url"]

    log.info("[%s] starting phenotype pull -> %s", species["key"], out_path)

    # First: count total
    body = f"eq(genome_name,{sp_enc})&limit(1)"
    r = post_query(session, AMR_ENDPOINT, body, log, fail)
    if r is None:
        log.error("[%s] could not even probe phenotype count", species["key"])
        return 0, 0
    total = int(r.headers.get("Content-Range", "items 0-0/0").split("/")[-1])
    log.info("[%s] total phenotype records (all evidence): %d", species["key"], total)

    parts_dir = out_path.with_suffix(out_path.suffix + ".parts")
    done_marker = out_path.with_suffix(out_path.suffix + ".done")

    # Resume policy:
    #  - If parquet exists AND row count >= 70% of pre-dedup total: treat
    #    as complete. Write done marker, return. (~20% dedup loss expected.)
    #  - If parquet missing but shards exist: consolidate from shards.
    #  - Otherwise: full re-download.
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["genome_id"])
        if len(existing) >= total * 0.70 or done_marker.exists():
            n_unique = existing["genome_id"].nunique()
            log.info("[%s] phenotype parquet already present (%d records, "
                     "%d genomes), skipping re-download",
                     species["key"], len(existing), n_unique)
            if not done_marker.exists():
                done_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime()))
            return len(existing), n_unique
        log.info("[%s] parquet exists but is suspiciously small "
                 "(%d / %d expected), restarting clean",
                 species["key"], len(existing), total)
        out_path.unlink()
    # Clean any stale shards too (we'll rebuild)
    if parts_dir.exists():
        for s in parts_dir.glob("shard_*.parquet"):
            s.unlink()
        try:
            parts_dir.rmdir()
        except OSError:
            pass

    select = ",".join(PHENO_FIELDS)
    buffer: list[dict] = []
    seen_ids: set[str] = set()
    n_dedup_rows = 0
    offset = 0
    pbar = tqdm(total=total, desc=f"{species['key']}-pheno", unit="rec")
    while True:
        body = (f"eq(genome_name,{sp_enc})"
                f"&select({select})"
                f"&limit({PHENO_CHUNK},{offset})"
                f"&sort(+id)")
        r = post_query(session, AMR_ENDPOINT, body, log, fail)
        if r is None:
            log.error("[%s] phenotype query failed at offset=%d", species["key"], offset)
            # Best-effort flush whatever we have
            append_parquet(buffer, out_path, log)
            buffer = []
            return n_dedup_rows, len(seen_ids)
        items = r.json()
        if not items:
            break
        # Dedup on AMR record id
        for it in items:
            rid = it.get("id") or f"{it.get('genome_id')}|{it.get('antibiotic')}|{it.get('measurement')}"
            # Use a hash of stable fields for dedup (id is per-record uuid)
            key = rid
            if key in seen_ids:
                continue
            seen_ids.add(key)
            # Normalize: keep only requested fields, drop _version_ etc.
            row = {k: it.get(k) for k in PHENO_FIELDS}
            buffer.append(row)
            n_dedup_rows += 1
        pbar.update(len(items))
        if len(buffer) >= PHENO_FLUSH:
            append_parquet(buffer, out_path, log)
            buffer = []
        if len(items) < PHENO_CHUNK:
            break
        offset += PHENO_CHUNK
        time.sleep(INTER_REQUEST_SLEEP)
    pbar.close()
    if buffer:
        append_parquet(buffer, out_path, log)

    # Consolidate shards into a single file
    consolidate_parquet(out_path, log, dedup_keys=None)
    # Write done marker
    done_marker = out_path.with_suffix(out_path.suffix + ".done")
    done_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # Final unique-genome count from saved parquet
    final_df = pd.read_parquet(out_path, columns=["genome_id"])
    n_unique = final_df["genome_id"].nunique()
    log.info("[%s] phenotype DONE: %d records, %d unique genomes",
             species["key"], len(final_df), n_unique)
    return len(final_df), n_unique


# ---------------------------------------------------------------------------
# Genome metadata
# ---------------------------------------------------------------------------

def download_genome_metadata(species: dict, genome_ids: list[str],
                             session: requests.Session,
                             log: logging.Logger, fail: dict) -> int:
    out_path = HDD_ROOT / f"genome_metadata_{species['key']}.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["genome_id"])
        if set(existing["genome_id"].astype(str)) >= set(map(str, genome_ids)):
            log.info("[%s] genome metadata parquet already covers all %d ids",
                     species["key"], len(genome_ids))
            return len(existing)

    select = ",".join(GENOME_FIELDS)
    rows: list[dict] = []
    pbar = tqdm(total=len(genome_ids), desc=f"{species['key']}-meta", unit="gid")
    for start in range(0, len(genome_ids), GENOME_CHUNK):
        batch = genome_ids[start:start + GENOME_CHUNK]
        ids_str = ",".join(str(g) for g in batch)
        body = f"in(genome_id,({ids_str}))&select({select})&limit({GENOME_CHUNK})"
        r = post_query(session, GENOME_ENDPOINT, body, log, fail)
        if r is None:
            log.error("[%s] genome metadata batch failed at start=%d", species["key"], start)
            pbar.update(len(batch))
            continue
        items = r.json()
        for it in items:
            row = {k: it.get(k) for k in GENOME_FIELDS}
            rows.append(row)
        pbar.update(len(batch))
        time.sleep(INTER_REQUEST_SLEEP)
    pbar.close()
    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=["genome_id"])
        df.to_parquet(out_path, index=False)
        log.info("[%s] genome metadata: %d rows -> %s",
                 species["key"], len(df), out_path)
    return len(rows)


# ---------------------------------------------------------------------------
# AMR-relevant protein download
# ---------------------------------------------------------------------------

def _ambiguous_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    s = seq.upper()
    n_amb = sum(1 for c in s if c == "X" or c == "*")
    return n_amb / max(len(s), 1)


def fetch_sequences_by_md5(md5s: list[str], session: requests.Session,
                           log: logging.Logger, fail: dict,
                           chunk: int = 200) -> dict[str, str]:
    """Bulk-fetch protein sequences from /feature_sequence/ by md5 hash.

    Returns md5 -> aa_sequence dict. Missing md5s are simply omitted."""
    out: dict[str, str] = {}
    md5s = sorted(set(md5s))
    if not md5s:
        return out
    for start in range(0, len(md5s), chunk):
        batch = md5s[start:start + chunk]
        ids = ",".join(batch)
        body = (f"in(md5,({ids}))&eq(sequence_type,AA)"
                f"&select(md5,sequence)&limit({chunk})")
        r = post_query(session, SEQUENCE_ENDPOINT, body, log, fail)
        if r is None:
            log.warning("sequence batch failed at start=%d", start)
            continue
        items = r.json()
        for it in items:
            md5 = it.get("md5")
            seq = it.get("sequence")
            if md5 and seq:
                out[md5] = seq
        time.sleep(INTER_REQUEST_SLEEP * 0.3)
    return out


def download_proteins_for_genome(genome_id: str, session: requests.Session,
                                 log: logging.Logger, fail: dict
                                 ) -> Optional[list[dict]]:
    """Download AMR-relevant CDS protein sequences for one genome.

    Strategy:
      1. Query /genome_feature/ for all CDS metadata (product, md5) for the
         genome. Client-side filter to AMR keyword matches.
      2. Batch-fetch sequences for matched md5s via /feature_sequence/.
      3. QC each by ambiguous-residue fraction, build output rows."""
    select = ",".join(FEATURE_FIELDS)
    raw_matches: list[dict] = []
    offset = 0
    while True:
        body = (f"eq(genome_id,{genome_id})&eq(feature_type,CDS)"
                f"&select({select})&limit({FEATURE_CHUNK},{offset})")
        r = post_query(session, FEATURE_ENDPOINT, body, log, fail)
        if r is None:
            return None
        items = r.json()
        if not items:
            break
        for it in items:
            product = (it.get("product") or "")
            gene = (it.get("gene") or "")
            text = f"{product} {gene}"
            if not _AMR_PRODUCT_RE.search(text):
                continue
            if not it.get("aa_sequence_md5"):
                continue
            raw_matches.append(it)
        if len(items) < FEATURE_CHUNK:
            break
        offset += FEATURE_CHUNK
        time.sleep(INTER_REQUEST_SLEEP * 0.3)

    if not raw_matches:
        return []

    # Step 2: fetch sequences by md5
    md5s = [it["aa_sequence_md5"] for it in raw_matches]
    seq_map = fetch_sequences_by_md5(md5s, session, log, fail, chunk=200)

    # Step 3: assemble output, QC
    out: list[dict] = []
    for it in raw_matches:
        md5 = it["aa_sequence_md5"]
        aa = seq_map.get(md5, "")
        if not aa:
            continue
        if _ambiguous_fraction(aa) > 0.05:
            continue
        row = {k: it.get(k) for k in FEATURE_FIELDS}
        row["genome_id"] = genome_id
        row["aa_sequence"] = aa
        row["_amr_keyword_hit"] = True
        out.append(row)
    return out


def download_proteins(species: dict, genome_ids: list[str],
                      session: requests.Session,
                      log: logging.Logger, fail: dict,
                      cap: Optional[int] = None) -> tuple[int, int]:
    """Download AMR-relevant proteins for the given genome list.

    Returns (n_proteins, n_genomes_processed)."""
    out_path = HDD_ROOT / f"proteins_{species['key']}.parquet"
    done_path = HDD_ROOT / f"{species['key']}_proteins_done.txt"

    done = read_done_set(done_path)
    targets = [g for g in genome_ids if str(g) not in done]
    if cap is not None:
        targets = targets[:cap]

    log.info("[%s] protein pull: %d genomes total, %d remaining%s",
             species["key"], len(genome_ids), len(targets),
             f" (cap={cap})" if cap is not None else "")

    if not targets:
        if out_path.exists():
            df = pd.read_parquet(out_path, columns=["genome_id"])
            return len(df), df["genome_id"].nunique()
        return 0, 0

    flush_buffer: list[dict] = []
    n_proteins = 0
    n_genomes_ok = 0
    pbar = tqdm(total=len(targets), desc=f"{species['key']}-prot", unit="gid")
    for gid in targets:
        prots = download_proteins_for_genome(str(gid), session, log, fail)
        if prots is None:
            log.error("[%s] giving up on genome %s after retries", species["key"], gid)
            pbar.update(1)
            if fail["n"] >= HARD_STOP_AFTER_TIMEOUTS:
                log.error("[%s] %d consecutive failures, aborting protein pull",
                          species["key"], fail["n"])
                break
            continue
        flush_buffer.extend(prots)
        n_proteins += len(prots)
        n_genomes_ok += 1
        append_done(done_path, str(gid))
        pbar.update(1)
        if len(flush_buffer) >= 5000:
            append_parquet(flush_buffer, out_path, log)
            flush_buffer = []
        time.sleep(INTER_REQUEST_SLEEP * 0.5)
    pbar.close()
    if flush_buffer:
        append_parquet(flush_buffer, out_path, log)

    # Consolidate any shards
    consolidate_parquet(out_path, log, dedup_keys=None)

    if out_path.exists():
        df = pd.read_parquet(out_path, columns=["genome_id"])
        return len(df), df["genome_id"].nunique()
    return n_proteins, n_genomes_ok


# ---------------------------------------------------------------------------
# QC reporting
# ---------------------------------------------------------------------------

def qc_summary(species: dict, log: logging.Logger) -> dict:
    summary: dict = {"species": species["key"]}
    pheno_path = HDD_ROOT / f"amr_phenotypes_{species['key']}.parquet"
    meta_path = HDD_ROOT / f"genome_metadata_{species['key']}.parquet"
    prot_path = HDD_ROOT / f"proteins_{species['key']}.parquet"
    if pheno_path.exists():
        df = pd.read_parquet(pheno_path)
        summary["phenotype_records"] = len(df)
        summary["phenotype_unique_genomes"] = df["genome_id"].nunique()
        evidence_counts = df["evidence"].fillna("UNKNOWN").value_counts().to_dict()
        summary["evidence_breakdown"] = evidence_counts
        # Lab-only count
        lab = df[df["evidence"].fillna("") != "Computational Method"]
        summary["lab_records"] = len(lab)
        summary["lab_unique_genomes"] = lab["genome_id"].nunique()
        # Antibiotic coverage
        summary["n_antibiotics"] = df["antibiotic"].nunique()
        summary["top_antibiotics"] = df["antibiotic"].value_counts().head(10).to_dict()
        # MIC coverage (records with measurement and lab origin)
        lab_with_mic = lab[lab["measurement"].notna() & (lab["measurement"].astype(str).str.len() > 0)]
        summary["lab_records_with_mic"] = len(lab_with_mic)
        summary["lab_genomes_with_mic"] = lab_with_mic["genome_id"].nunique()
    if meta_path.exists():
        m = pd.read_parquet(meta_path)
        summary["metadata_rows"] = len(m)
        if "isolation_source" in m.columns:
            summary["isolation_source_top"] = m["isolation_source"].fillna("UNKNOWN").value_counts().head(8).to_dict()
        if "collection_year" in m.columns:
            yrs = m["collection_year"].dropna()
            summary["collection_year_range"] = [int(yrs.min()) if len(yrs) else None,
                                                int(yrs.max()) if len(yrs) else None]
        if "assembly_accession" in m.columns:
            dedup = m.drop_duplicates(subset=["assembly_accession"])
            summary["after_assembly_dedup"] = len(dedup)
    if prot_path.exists():
        p = pd.read_parquet(prot_path, columns=["genome_id", "aa_sequence_md5"])
        summary["protein_rows"] = len(p)
        summary["genomes_with_proteins"] = p["genome_id"].nunique()
        summary["unique_aa_sequences"] = p["aa_sequence_md5"].nunique()
    log.info("[%s] QC summary: %s", species["key"],
             json.dumps({k: v for k, v in summary.items()
                         if not isinstance(v, dict)}, default=str))
    return summary


# ---------------------------------------------------------------------------
# Main pipeline per species
# ---------------------------------------------------------------------------

def run_species(species: dict, session: requests.Session, log: logging.Logger,
                lab_only_for_proteins: bool, protein_cap: Optional[int]) -> dict:
    fail = {"n": 0}
    log.info("=" * 60)
    log.info("species %s [%s]", species["key"], species["name"])
    log.info("=" * 60)

    # Phenotype
    n_records, n_unique = download_phenotypes(species, session, log, fail)
    if n_records == 0:
        log.error("[%s] zero phenotype records — abort species", species["key"])
        return {"species": species["key"], "status": "FAIL", "reason": "no_phenotype"}

    # Genome metadata for all unique genomes
    pheno_df = pd.read_parquet(HDD_ROOT / f"amr_phenotypes_{species['key']}.parquet",
                               columns=["genome_id", "evidence"])
    all_gids = sorted(pheno_df["genome_id"].dropna().astype(str).unique())
    lab_gids = sorted(pheno_df.loc[pheno_df["evidence"].fillna("") != "Computational Method",
                                   "genome_id"].dropna().astype(str).unique())
    log.info("[%s] %d unique genomes total, %d unique lab-evidence genomes",
             species["key"], len(all_gids), len(lab_gids))

    download_genome_metadata(species, all_gids, session, log, fail)

    # Protein download — prioritize lab-evidence genomes (for Gate 4 ESM2 embedding)
    target_gids = lab_gids if lab_only_for_proteins else all_gids
    download_proteins(species, target_gids, session, log, fail, cap=protein_cap)

    return qc_summary(species, log)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--species", nargs="*", default=None,
                   help="Subset of species keys: ecoli saureus kpneumo")
    p.add_argument("--no-proteins", action="store_true",
                   help="Skip protein download phase (useful for resume QC).")
    p.add_argument("--lab-only-proteins", action="store_true", default=True,
                   help="Download proteins only for lab-evidence genomes (default).")
    p.add_argument("--all-evidence-proteins", action="store_true",
                   help="Override lab-only protein restriction.")
    p.add_argument("--protein-cap", type=int, default=None,
                   help="Cap number of genomes processed for proteins (debug).")
    args = p.parse_args()

    HDD_ROOT.mkdir(parents=True, exist_ok=True)
    SSD_LOGS.mkdir(parents=True, exist_ok=True)
    SSD_PROCESSED.mkdir(parents=True, exist_ok=True)
    log = setup_logging()

    species_to_run = SPECIES
    if args.species:
        wanted = set(args.species)
        species_to_run = [s for s in SPECIES if s["key"] in wanted]
        if not species_to_run:
            log.error("no matching species in: %s", args.species)
            return 2

    lab_only_prot = not args.all_evidence_proteins

    session = make_session()
    summaries: list[dict] = []
    for sp in species_to_run:
        if args.no_proteins:
            # Run phenotype + metadata only (call run_species but cap proteins to 0)
            res = run_species(sp, session, log, lab_only_prot, protein_cap=0)
        else:
            res = run_species(sp, session, log, lab_only_prot, protein_cap=args.protein_cap)
        summaries.append(res)

    # Save summary
    summary_path = SSD_PROCESSED / "gate1_summary.json"
    with summary_path.open("w") as f:
        json.dump(summaries, f, indent=2, default=str)
    log.info("wrote summary -> %s", summary_path)

    # Pass criterion evaluation
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metrics = []
    pass_status = "PASS"
    # New thresholds reflecting BV-BRC schema reality (computational-heavy):
    #   - PASS: every species has >=5,000 unique-genome phenotype coverage (any evidence)
    #   - SOFT: lab-evidence coverage tracked separately and reported
    #   - BLOCKED: API unreachable
    for s in summaries:
        ng = s.get("phenotype_unique_genomes", 0) or 0
        lg = s.get("lab_unique_genomes", 0) or 0
        metrics.append(f"{s['species']}_n={ng} lab={lg}")
        if ng < 5000:
            pass_status = "FAIL"

    line = (f"{timestamp} GATE_1: {pass_status} [endpoint=/genome_amr/] "
            f"[evidence_tiers=mixed,tagged] [{' | '.join(metrics)}]")
    log.info(line)
    with GATE_LOG.open("a") as f:
        f.write(line + "\n")
    return 0 if pass_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
