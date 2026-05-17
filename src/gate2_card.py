"""
Gate 2 — CARD download + AMRFinderPlus annotation of Gate 1 genomes.

This script:
  1. Verifies CARD assets at /mnt/hdd/research/amr/raw/card/.
  2. Writes per-genome protein FASTAs from proteins_<sp>.parquet
     (resumable — skips if file already exists).
  3. Runs AMRFinderPlus in protein-only mode on each FASTA in parallel.
  4. Parses AMRFinderPlus TSVs into a single resistance_genes.parquet.
  5. Cross-references point mutations against CARD variant model catalogue.
  6. Computes the gate pass criterion: % of phenotype-resistant genomes
     (lab-evidence) with at least one AMR call.

Resumable: re-running picks up where it left off (per-genome FASTA and
AMRFinder TSV outputs both serve as resumable artefacts).
"""

import os
import sys
import json
import time
import logging
import multiprocessing as mp
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------- Paths ----------
ROOT          = Path("/mnt/ssd/research/amr")
HDD_RAW       = Path("/mnt/hdd/research/amr/raw")
CARD_DIR      = HDD_RAW / "card"
FASTA_DIR     = ROOT / "processed" / "protein_fastas"
CALLS_DIR     = ROOT / "processed" / "amrfinder_calls"
AMRFP_DB      = ROOT / "processed" / "amrfinderplus_db" / "latest"
LOG_DIR       = ROOT / "logs"
OUT_PARQUET   = ROOT / "processed" / "resistance_genes.parquet"
GATE_LOG      = LOG_DIR / "gates.log"
GATE2_LOG     = LOG_DIR / "gate2_card.log"
SUMMARY_JSON  = ROOT / "processed" / "gate2_summary.json"

AMRFINDER_BIN = "/home/rohit/miniconda3/envs/amrfinder/bin/amrfinder"

SPECIES = {
    "ecoli":   {"organism": "Escherichia",        "proteins_pq": HDD_RAW/"proteins_ecoli.parquet",
                "amr_pq":   HDD_RAW/"amr_phenotypes_ecoli.parquet"},
    "saureus": {"organism": "Staphylococcus_aureus","proteins_pq": HDD_RAW/"proteins_saureus.parquet",
                "amr_pq":   HDD_RAW/"amr_phenotypes_saureus.parquet"},
    "kpneumo": {"organism": "Klebsiella_pneumoniae","proteins_pq": HDD_RAW/"proteins_kpneumo.parquet",
                "amr_pq":   HDD_RAW/"amr_phenotypes_kpneumo.parquet"},
}

# Mutation-conferring resistance gene families (chromosomal + relevant plasmid)
TARGET_FAMILIES = {
    "BETA-LACTAM", "BLA", "BLAA", "BLAB", "BLAC", "BLAD",
    "QUINOLONE",
    "AMINOGLYCOSIDE",
    "GLYCOPEPTIDE",
    "COLISTIN",   # mcr genes — annotate but tag plasmid
    "TRIMETHOPRIM", "DIHYDROFOLATE",
    "TETRACYCLINE", "MACROLIDE", "PHENICOL", "RIFAMYCIN",
    "FOSFOMYCIN", "SULFONAMIDE",
}

# ---------- Logging ----------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.FileHandler(GATE2_LOG), logging.StreamHandler()],
)
log = logging.getLogger("gate2")


# ===========================================================================
# CARD variant catalogue: known resistance mutations per gene
# ===========================================================================
def parse_card_variant_mutations() -> Dict[str, set]:
    """
    Parse CARD protein_fasta_protein_variant_model.fasta headers to extract
    the (gene_name -> {known mutations}) catalogue.

    Header format typically:
      >gb|REF.1|ARO:NNNNNNN|name [Organism]
    with the mutation annotated in a companion file (snps.txt) OR embedded
    in the model name. We use snps.txt where available.
    """
    snps_path = CARD_DIR / "snps.txt"
    catalogue: Dict[str, set] = defaultdict(set)
    if not snps_path.exists():
        log.warning("snps.txt not present in CARD download — variant catalogue limited.")
        return catalogue

    # snps.txt format (observed):
    #   Accession  Name  Model Type  Parameter Type  Mutations  CARD Short Name  source  citation
    # where 'Mutations' is the mutation string (e.g. 'S81L') and
    # 'CARD Short Name' is like 'Abau_gyrA_FLO' — gene token is the 2nd '_'-split chunk.
    try:
        with snps_path.open() as f:
            header = f.readline().rstrip("\n").split("\t")
            idx = {c: i for i, c in enumerate(header)}
            mut_col   = idx.get("Mutations", 4)
            short_col = idx.get("CARD Short Name", 5)
            name_col  = idx.get("Name", 1)
            for line in f:
                cells = line.rstrip("\n").split("\t")
                if len(cells) <= max(mut_col, short_col):
                    continue
                mutation  = cells[mut_col].strip()
                short_name = cells[short_col].strip()  # e.g. 'Abau_gyrA_FLO'
                full_name = cells[name_col].strip()
                # Extract gene tokens from short_name: middle chunk after splitting on '_'
                gene_keys = set()
                if "_" in short_name:
                    parts = short_name.split("_")
                    if len(parts) >= 2:
                        gene_keys.add(parts[1].lower())  # 'gyra'
                # Also pull single-token gene name out of full Name (first capitalised word like 'gyrA')
                m = re.search(r"\b([a-z]{2,5}[A-Z])\b", full_name)
                if m:
                    gene_keys.add(m.group(1).lower())
                # mutation may be a single 'S81L' or a comma-list
                for mp in MUT_DELIM_SPLIT.split(mutation):
                    mp = mp.strip()
                    if re.match(r"^[A-Z]\d+[A-Z\*]$", mp):
                        for gk in gene_keys:
                            catalogue[gk].add(mp)
    except Exception as e:
        log.warning(f"Failed to parse snps.txt: {e}")

    log.info(f"CARD variant catalogue: {len(catalogue)} genes, "
             f"{sum(len(v) for v in catalogue.values())} known mutations")
    return catalogue


# ===========================================================================
# Per-genome FASTA writing (resumable)
# ===========================================================================
def write_fastas_for_species(sp: str) -> List[Tuple[str, Path]]:
    """Write per-genome protein FASTAs. Returns list of (genome_id, fasta_path)."""
    info = SPECIES[sp]
    fa_root = FASTA_DIR / sp
    fa_root.mkdir(parents=True, exist_ok=True)

    log.info(f"[{sp}] loading proteins parquet …")
    df = pd.read_parquet(info["proteins_pq"])
    df = df[df["aa_sequence"].notna() & (df["aa_sequence"].str.len() >= 10)]
    log.info(f"[{sp}] {df.shape[0]:,} protein rows across {df['genome_id'].nunique()} genomes")

    # FORCE_REWRITE: set env var to non-empty to overwrite all FASTAs
    # (used after a deduplication bug fix so previously written files are
    # rewritten with unique identifiers).
    force_rewrite = bool(os.environ.get("FORCE_REWRITE", ""))

    pairs: List[Tuple[str, Path]] = []
    grouped = df.groupby("genome_id", sort=False)
    for gid, sub in tqdm(grouped, desc=f"[{sp}] writing FASTAs", total=df["genome_id"].nunique()):
        fp = fa_root / f"{gid}.faa"
        pairs.append((gid, fp))
        if fp.exists() and fp.stat().st_size > 0 and not force_rewrite:
            continue
        # Dedupe per (genome, md5): if multiple CDS encode the same protein,
        # AMRFinderPlus's fasta_check rejects duplicate headers. We keep one
        # representative per md5, but track copy_number for downstream use.
        seen_md5: set = set()
        with fp.open("w") as fh:
            for _, r in sub.iterrows():
                md5 = r["aa_sequence_md5"]
                if md5 in seen_md5:
                    continue
                seen_md5.add(md5)
                aa = str(r["aa_sequence"]).rstrip("*").replace("\n", "")
                if len(aa) < 10:
                    continue
                product = str(r.get("product", "")).replace("\n", " ")[:200]
                fh.write(f">{md5} {product}\n{aa}\n")
    log.info(f"[{sp}] {len(pairs)} FASTAs available")
    return pairs


# ===========================================================================
# AMRFinderPlus invocation (one genome at a time, suitable for mp.Pool)
# ===========================================================================
def run_amrfinder_one(args) -> Tuple[str, str, Optional[str]]:
    """Run amrfinder on a single genome FASTA. Returns (sp, gid, error or None)."""
    sp, gid, fa_path, organism = args
    out_dir = CALLS_DIR / sp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{gid}.tsv"
    if out_path.exists() and out_path.stat().st_size > 0:
        return (sp, gid, None)

    cmd = [
        AMRFINDER_BIN,
        "-p", str(fa_path),
        "-d", str(AMRFP_DB),
        "-O", organism,
        "--threads", "1",
        "--quiet",
        "-o", str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=600,
        )
        if proc.returncode != 0:
            # Some genomes may legitimately produce no calls — that's fine.
            # But non-zero return = real failure.
            err_msg = proc.stderr.decode("utf-8", errors="ignore")[:500]
            if out_path.exists():
                out_path.unlink()
            return (sp, gid, f"rc={proc.returncode}: {err_msg}")
    except subprocess.TimeoutExpired:
        return (sp, gid, "timeout")
    except Exception as e:
        return (sp, gid, f"exc: {e}")
    return (sp, gid, None)


def run_all(pairs_by_sp: Dict[str, List[Tuple[str, Path]]], n_workers: int = 4):
    """Run AMRFinderPlus across all genomes from all species, in parallel."""
    work: List[Tuple[str, str, Path, str]] = []
    for sp, pairs in pairs_by_sp.items():
        org = SPECIES[sp]["organism"]
        for gid, fp in pairs:
            out_path = CALLS_DIR / sp / f"{gid}.tsv"
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            work.append((sp, gid, fp, org))

    log.info(f"AMRFinderPlus queue: {len(work)} genomes to process (parallel={n_workers})")
    if not work:
        return

    errors = []
    with mp.Pool(n_workers) as pool:
        for sp, gid, err in tqdm(pool.imap_unordered(run_amrfinder_one, work),
                                  total=len(work), desc="AMRFinder"):
            if err is not None:
                errors.append((sp, gid, err))
                if len(errors) <= 5:
                    log.warning(f"[{sp}/{gid}] {err}")

    log.info(f"AMRFinderPlus done. Errors: {len(errors)} of {len(work)}")
    if errors:
        err_path = LOG_DIR / "gate2_amrfinder_errors.tsv"
        with err_path.open("w") as fh:
            fh.write("species\tgenome_id\terror\n")
            for sp, gid, err in errors:
                fh.write(f"{sp}\t{gid}\t{err}\n")
        log.info(f"Error log: {err_path}")


# ===========================================================================
# Parse AMRFinderPlus TSVs and aggregate
# ===========================================================================
AMRFP_COLS_KEEP = [
    "Protein id", "Element symbol", "Element name", "Scope", "Type", "Subtype",
    "Class", "Subclass", "Method", "% Coverage of reference",
    "% Identity to reference", "Closest reference accession",
    "Closest reference name", "HMM accession", "HMM description",
]

MUT_RE = re.compile(r"\b([A-Z])(\d+)([A-Z\*])\b")
MUT_DELIM_SPLIT = re.compile(r"[,;|_\s]+")


def extract_point_mutations(symbol: str, name: str, method: str) -> List[str]:
    """
    From AMRFinderPlus call rows of type 'POINT' or with 'POINTX' methods,
    extract amino-acid substitutions like 'S83L'. Falls back to scanning
    the element name string.
    """
    # POINT method elements typically encode mutation in the symbol, e.g. "gyrA_S83L"
    muts: List[str] = []
    for src in (symbol, name):
        if not isinstance(src, str):
            continue
        for token in MUT_DELIM_SPLIT.split(src):
            for m in MUT_RE.finditer(token):
                wt, pos, alt = m.group(1), int(m.group(2)), m.group(3)
                # avoid catching things like 'D5' from accessions; require >=2-digit position OR known gene context
                muts.append(f"{wt}{pos}{alt}")
    # dedupe while preserving order
    seen = set(); out = []
    for x in muts:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def parse_calls(genomes_by_sp: Dict[str, List[str]],
                pheno_evidence: Dict[Tuple[str, str], str]) -> pd.DataFrame:
    rows = []
    for sp, gids in genomes_by_sp.items():
        for gid in tqdm(gids, desc=f"[{sp}] parsing TSVs"):
            tsv = CALLS_DIR / sp / f"{gid}.tsv"
            if not tsv.exists() or tsv.stat().st_size == 0:
                continue
            try:
                df = pd.read_csv(tsv, sep="\t", dtype=str)
            except Exception:
                continue
            if df.empty:
                continue
            evid = pheno_evidence.get((sp, gid), "unknown")
            for _, r in df.iterrows():
                symbol = r.get("Element symbol", "") or ""
                name   = r.get("Element name", "") or ""
                method = r.get("Method", "") or ""
                muts   = extract_point_mutations(symbol, name, method) \
                         if method.startswith("POINT") else []
                try:
                    ident = float(r.get("% Identity to reference", "") or 0.0)
                except ValueError:
                    ident = 0.0
                try:
                    cov   = float(r.get("% Coverage of reference", "") or 0.0)
                except ValueError:
                    cov = 0.0
                rows.append({
                    "genome_id":      gid,
                    "species":        sp,
                    "gene_symbol":    symbol,
                    "gene_family":    r.get("Element name", ""),
                    "mechanism":      r.get("Subtype", ""),
                    "identity_pct":   ident,
                    "coverage_pct":   cov,
                    "amr_class":      r.get("Class", ""),
                    "amr_subclass":   r.get("Subclass", ""),
                    "method":         method,
                    "closest_ref":    r.get("Closest reference accession", ""),
                    "hmm_accession":  r.get("HMM accession", ""),
                    "point_mutations": muts,
                    "evidence_tier":  evid,
                    "is_plasmid_borne": symbol.lower().startswith("mcr"),
                })
    return pd.DataFrame(rows)


# ===========================================================================
# Phenotype helper
# ===========================================================================
def load_pheno_resistance(sp: str) -> Tuple[set, Dict[Tuple[str, str], str]]:
    """
    Returns:
      - set of genome_ids that have ANY lab-evidence resistant/non-susceptible
        phenotype call (for our annotated genome subset).
      - per-genome evidence tier mapping (sp, gid) -> "lab" | "computational"
        | "mixed".
    """
    info = SPECIES[sp]
    df = pd.read_parquet(info["amr_pq"], columns=["genome_id", "resistant_phenotype", "evidence"])
    # Normalize genome_id to string (BV-BRC uses float-like ids; parquet may store either)
    df["genome_id"] = df["genome_id"].astype(str)

    resistant_mask = df["resistant_phenotype"].isin(["Resistant", "Non-susceptible", "Nonsusceptible"])
    lab_mask = df["evidence"] == "Laboratory Method"
    lab_resistant = set(df.loc[resistant_mask & lab_mask, "genome_id"].unique().tolist())

    # evidence tier per genome (lab if any lab row exists; else computational)
    evid_map: Dict[Tuple[str, str], str] = {}
    by_gid = df.groupby("genome_id")["evidence"].apply(set)
    for gid, evs in by_gid.items():
        if "Laboratory Method" in evs:
            evid_map[(sp, gid)] = "lab"
        elif "Computational Method" in evs:
            evid_map[(sp, gid)] = "computational"
        else:
            evid_map[(sp, gid)] = "unknown"
    return lab_resistant, evid_map


# ===========================================================================
# Cross-reference point mutations against CARD catalogue
# ===========================================================================
def cross_reference_mutations(df: pd.DataFrame, catalogue: Dict[str, set]) -> pd.DataFrame:
    """
    For each row with point_mutations, flag whether any of them is in CARD's
    known-resistance catalogue for that gene (matched by short name token).
    """
    def check(row):
        muts = row.get("point_mutations") or []
        if not muts:
            return False
        symbol = (row.get("gene_symbol") or "").lower()
        # gene root: strip suffix like "_S83L"
        root = symbol.split("_")[0] if "_" in symbol else symbol
        cat_for_gene = catalogue.get(root, set()) | catalogue.get(symbol, set())
        if not cat_for_gene:
            # try every catalogue gene that the symbol contains
            for k in catalogue:
                if k and (k in symbol or symbol in k):
                    cat_for_gene |= catalogue[k]
        return any(m in cat_for_gene for m in muts)

    df["known_resistance_mutation"] = df.apply(check, axis=1)
    return df


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    log.info("=" * 80)
    log.info("GATE 2 — CARD + AMRFinderPlus annotation")
    log.info(f"CARD dir:      {CARD_DIR}")
    log.info(f"AMRFP DB:      {AMRFP_DB.resolve()}")
    log.info(f"FASTA dir:     {FASTA_DIR}")
    log.info(f"Calls dir:     {CALLS_DIR}")
    log.info(f"Output parquet: {OUT_PARQUET}")
    log.info("=" * 80)

    # 1. CARD catalogue
    catalogue = parse_card_variant_mutations()

    # 2. Per-genome FASTAs
    pairs_by_sp: Dict[str, List[Tuple[str, Path]]] = {}
    for sp in SPECIES:
        pairs_by_sp[sp] = write_fastas_for_species(sp)

    # 3. Run AMRFinderPlus
    n_workers = int(os.environ.get("AMRFP_WORKERS", "4"))
    run_all(pairs_by_sp, n_workers=n_workers)

    # 4. Parse calls — but first gather pheno evidence maps
    genomes_by_sp: Dict[str, List[str]] = {sp: [g for g, _ in pairs] for sp, pairs in pairs_by_sp.items()}
    pheno_evidence: Dict[Tuple[str, str], str] = {}
    lab_resistant_by_sp: Dict[str, set] = {}
    log.info("Loading phenotype evidence maps …")
    for sp in SPECIES:
        lr, em = load_pheno_resistance(sp)
        lab_resistant_by_sp[sp] = lr
        pheno_evidence.update(em)
        log.info(f"[{sp}] lab-evidence resistant genomes (entire phenotype table): {len(lr):,}")

    log.info("Parsing AMRFinderPlus TSVs into a unified table …")
    calls_df = parse_calls(genomes_by_sp, pheno_evidence)
    log.info(f"Raw call rows: {len(calls_df):,}")

    # 5. Cross-reference with CARD catalogue
    calls_df = cross_reference_mutations(calls_df, catalogue)

    # 6. Save
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    calls_df.to_parquet(OUT_PARQUET, index=False)
    log.info(f"Wrote {OUT_PARQUET}")

    # 7. Gate criterion: % phenotype-resistant (lab) genomes with ≥1 call
    summary: Dict = {"species": {}}
    overall_resistant_annotated = 0
    overall_resistant_total = 0
    total_genomes_processed = 0
    total_genomes_with_call = 0

    for sp, gids in genomes_by_sp.items():
        gids_set = set(gids)
        lab_resistant = lab_resistant_by_sp[sp] & gids_set
        sp_calls = calls_df[calls_df["species"] == sp]
        gids_with_call = set(sp_calls["genome_id"].unique().tolist())

        lab_res_with_call = lab_resistant & gids_with_call

        pct_call = (len(lab_res_with_call) / max(1, len(lab_resistant))) * 100.0

        summary["species"][sp] = {
            "n_genomes_processed":       len(gids),
            "n_genomes_with_any_call":   len(gids_with_call),
            "pct_genomes_with_any_call": round(len(gids_with_call) / max(1, len(gids)) * 100.0, 2),
            "n_phenotype_lab_resistant_in_processed":  len(lab_resistant),
            "n_phenotype_lab_resistant_with_call":     len(lab_res_with_call),
            "pct_phenotype_resistant_with_call":       round(pct_call, 2),
        }
        log.info(f"[{sp}] processed={len(gids)}, with_call={len(gids_with_call)}, "
                 f"lab_resistant_in_set={len(lab_resistant)}, lab_resistant_with_call={len(lab_res_with_call)} "
                 f"({pct_call:.1f}%)")

        overall_resistant_annotated += len(lab_res_with_call)
        overall_resistant_total += len(lab_resistant)
        total_genomes_processed += len(gids)
        total_genomes_with_call += len(gids_with_call)

    overall_pct = (overall_resistant_annotated / max(1, overall_resistant_total)) * 100.0

    # 8. Gene-family / mutation breakdowns for the report
    top_families = (calls_df["gene_symbol"].value_counts().head(20)).to_dict()
    class_dist   = (calls_df["amr_class"].value_counts().head(20)).to_dict()
    pm_df = calls_df[calls_df["point_mutations"].apply(lambda x: bool(x and len(x) > 0))]
    n_known   = int(pm_df["known_resistance_mutation"].sum())
    n_novel   = int(len(pm_df) - n_known)

    # Cross-evidence-tier sanity
    evid_call_rate = {}
    for tier in ["lab", "computational", "mixed", "unknown"]:
        gids_in_tier = {gid for (sp, gid), t in pheno_evidence.items() if t == tier}
        # restrict to processed genomes
        processed_set = set()
        for sp, gids in genomes_by_sp.items():
            processed_set |= set(gids)
        in_tier_processed = gids_in_tier & processed_set
        with_call = in_tier_processed & set(calls_df["genome_id"].unique().tolist())
        evid_call_rate[tier] = {
            "n_processed":      len(in_tier_processed),
            "n_with_call":      len(with_call),
            "call_rate_pct":    round(len(with_call) / max(1, len(in_tier_processed)) * 100.0, 2),
        }

    summary["overall"] = {
        "total_genomes_processed":          total_genomes_processed,
        "total_genomes_with_call":          total_genomes_with_call,
        "overall_pct_genomes_with_call":    round(total_genomes_with_call / max(1, total_genomes_processed) * 100.0, 2),
        "overall_lab_resistant":            overall_resistant_total,
        "overall_lab_resistant_with_call":  overall_resistant_annotated,
        "overall_pct_lab_resistant_with_call": round(overall_pct, 2),
    }
    summary["top_element_symbols"]   = top_families
    summary["amr_class_distribution"]= class_dist
    summary["point_mutations"]       = {
        "n_call_rows_with_point_mutations": int(len(pm_df)),
        "n_known_resistance":               n_known,
        "n_novel_or_unmapped":              n_novel,
    }
    summary["evidence_tier_call_rate"] = evid_call_rate
    summary["card_catalogue_size"]   = {
        "genes":     len(catalogue),
        "mutations": sum(len(v) for v in catalogue.values()),
    }
    summary["amrfinder_version"]   = "4.2.7"
    summary["amrfinder_db_version"]= "2026-03-24.1"
    summary["card_release"]        = "broadstreet_2025-05-29"
    summary["wall_clock_seconds"]  = round(time.time() - t0, 1)

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote summary {SUMMARY_JSON}")

    # 9. Gate decision
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if overall_pct >= 60.0:
        verdict = "PASS"
    elif overall_pct >= 40.0:
        verdict = "PASS_SOFT_FLAG"
    else:
        verdict = "HARD_STOP"

    gate_line = (
        f"{timestamp} GATE_2: {verdict} "
        f"[overall_pct_resistant_with_call={overall_pct:.2f}%] "
        f"[ecoli={summary['species']['ecoli']['pct_phenotype_resistant_with_call']}%] "
        f"[saureus={summary['species']['saureus']['pct_phenotype_resistant_with_call']}%] "
        f"[kpneumo={summary['species']['kpneumo']['pct_phenotype_resistant_with_call']}%] "
        f"[amrfp_db=2026-03-24.1] [card=2025-05-29]"
    )
    with GATE_LOG.open("a") as fh:
        fh.write(gate_line + "\n")
    log.info(gate_line)
    print(gate_line)

    log.info(f"GATE 2 verdict: {verdict}. Wall-clock: {summary['wall_clock_seconds']}s.")
    return verdict, summary


if __name__ == "__main__":
    v, s = main()
    sys.exit(0 if v.startswith("PASS") else 2)
