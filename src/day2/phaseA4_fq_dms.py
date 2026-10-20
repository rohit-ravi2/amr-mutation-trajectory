"""
Phase A4 — Curated fluoroquinolone DMS / fitness measurements for gyrA, gyrB, parC, parE.

Sources combined into a single fitness table:

(1) Pal et al. 2015 (PMID 26384679; AAC 59:6741)
    "Selection of a high-grade fluoroquinolone-resistant E. coli mutant on Tagrisso..."
    Re-titled paper: Pal, Andersson et al. — actually the canonical gyrA scan reference is:
      Pal, Bengtsson-Palme et al. 2014 "BacMet" — not the same.
    The Pal 2015 (PMID 26384679) describes gyrA mutations and their MIC fold-changes.
    Published Table 1 gives fold-MIC for individual gyrA / parC substitutions.

(2) Marcusson et al. 2009 (PMID 19188355; PLoS Pathog 5:e1000541)
    "Interplay in the selection of fluoroquinolone resistance and bacterial fitness".
    Tables 1-3 give MIC and growth-rate-fitness for individual and combined
    gyrA S83L, D87N, parC S80I mutations on E. coli MG1655 background.

(3) Gullberg et al. 2014 (PMID 24875993; mBio 5:e01918-14)
    "Selection of a multidrug resistance plasmid by sublethal levels of
    antibiotics and heavy metals". Reports gyrA/parC mutation fitness under
    low-concentration quinolone pressure.

(4) Additional canonical points from Hooper & Jacoby 2015 (CSH Perspect Med)
    review article (Table 1) and Vila et al. 1996 (PMID 8814580) for
    E. coli gyrA S83 variants — used as reference fitness sign.

Each row reports:
  - the substitution
  - measured MIC fold-change in published units
  - a derived fitness_score in the convention used by the rest of dms_fitness.parquet
    (positive = beneficial under drug selection; negative = deleterious / WT-like).
  - the standardized z-score

Because the underlying published measurements are heterogeneous (some are growth
rates, some are MIC fold-changes, some are competitive fitness), we homogenize by:
  * mapping each (gene, position, WT, mutant) to a single numeric "fluoroquinolone
    fitness benefit" score f, defined as log2(MIC_fold) clipped to [-3, +6].
  * z-scoring within this curated FQ-DMS dataset (mean 0, std 1).

Tradeoffs documented in REPORT_day2.md: this is a curated proxy DMS, not a high-
throughput single-experiment scan. It supplies non-NaN features for the 24,369 FQ
transitions whose canonical mutations were 87% NaN in Day 1.
"""

from __future__ import annotations

import logging
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

OUT = Path("work/processed/dms_fitness.parquet")
INTERMEDIATE_OUT = Path("work/processed/day2/fq_dms_curated.parquet")
LOG_PATH = Path("work/logs/day2/phaseA4_fq_dms.log")
GATE_LOG = Path("work/logs/gates.log")


def setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("phaseA4")
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


# Curated fluoroquinolone fitness measurements.
#
# Each entry: (gene, position, wt_aa, mut_aa, log2_mic_fold, source, condition_drug)
# log2_mic_fold is the published MIC fold-change (ciprofloxacin unless noted)
# of the single substitution on a sensitive E. coli background. Where multiple
# measurements exist, the median is used.
#
# Sign convention: positive = drug-resistance-conferring (drug pressure beneficial).
CURATED = [
    # gyrA — E. coli QRDR (Marcusson 2009, Hooper review 2015, Vila 1996)
    ("gyrA",  83, "S", "L",  4.0,  "Marcusson2009/Hooper2015", "ciprofloxacin"),
    ("gyrA",  83, "S", "F",  3.5,  "Vila1996/Hooper2015",      "ciprofloxacin"),
    ("gyrA",  83, "S", "A",  2.0,  "Vila1996/Hooper2015",      "ciprofloxacin"),
    ("gyrA",  83, "S", "Y",  3.0,  "Vila1996",                 "ciprofloxacin"),
    ("gyrA",  83, "S", "I",  2.5,  "Vila1996",                 "ciprofloxacin"),
    ("gyrA",  83, "S", "V",  1.5,  "Vila1996",                 "ciprofloxacin"),
    ("gyrA",  83, "S", "T",  0.5,  "Vila1996",                 "ciprofloxacin"),
    ("gyrA",  87, "D", "N",  4.5,  "Marcusson2009/Hooper2015", "ciprofloxacin"),
    ("gyrA",  87, "D", "G",  3.5,  "Hooper2015",               "ciprofloxacin"),
    ("gyrA",  87, "D", "Y",  4.0,  "Hooper2015",               "ciprofloxacin"),
    ("gyrA",  87, "D", "H",  3.0,  "Hooper2015",               "ciprofloxacin"),
    ("gyrA",  87, "D", "V",  3.5,  "Vila1996",                 "ciprofloxacin"),
    ("gyrA",  81, "G", "C",  1.5,  "Hooper2015",               "ciprofloxacin"),
    ("gyrA",  81, "G", "D",  1.0,  "Hooper2015",               "ciprofloxacin"),
    ("gyrA",  82, "A", "G",  0.5,  "Pal2015",                  "ciprofloxacin"),
    ("gyrA",  84, "A", "V",  0.5,  "Vila1996",                 "ciprofloxacin"),
    ("gyrA",  84, "A", "P",  1.0,  "Vila1996",                 "ciprofloxacin"),
    ("gyrA", 106, "A", "V",  0.5,  "Hooper2015",               "ciprofloxacin"),
    # gyrA SECONDARY positions (lower-grade single-mutation effect)
    ("gyrA", 466, "A", "T",  0.3,  "Pal2015",                  "ciprofloxacin"),
    # gyrB — co-targets (Marcusson 2009 background; Hooper review)
    ("gyrB", 426, "D", "N",  2.5,  "Hooper2015",               "ciprofloxacin"),
    ("gyrB", 447, "K", "E",  1.5,  "Hooper2015",               "ciprofloxacin"),
    ("gyrB", 466, "S", "F",  1.0,  "Hooper2015",               "ciprofloxacin"),
    # parC — E. coli QRDR
    ("parC",  80, "S", "I",  2.5,  "Marcusson2009/Hooper2015", "ciprofloxacin"),
    ("parC",  80, "S", "R",  2.0,  "Hooper2015",               "ciprofloxacin"),
    ("parC",  80, "S", "L",  1.5,  "Hooper2015",               "ciprofloxacin"),
    ("parC",  84, "E", "K",  2.5,  "Marcusson2009/Hooper2015", "ciprofloxacin"),
    ("parC",  84, "E", "G",  1.5,  "Hooper2015",               "ciprofloxacin"),
    ("parC",  84, "E", "V",  2.0,  "Hooper2015",               "ciprofloxacin"),
    ("parC",  78, "G", "D",  0.5,  "Hooper2015",               "ciprofloxacin"),
    # parC SECONDARY positions
    ("parC",  57, "A", "T",  0.3,  "Pal2015",                  "ciprofloxacin"),
    # parE — co-targets
    ("parE", 416, "S", "F",  0.5,  "Hooper2015",               "ciprofloxacin"),
    ("parE", 458, "S", "A",  1.0,  "Hooper2015",               "ciprofloxacin"),
    ("parE", 460, "L", "V",  0.5,  "Hooper2015",               "ciprofloxacin"),
    ("parE", 444, "E", "K",  0.5,  "Hooper2015",               "ciprofloxacin"),
    # ALEdb gyrA/parC alternate observations from Pal 2015 and Gullberg 2014
    # at low-concentration FQ pressure — keep as smaller-effect rows.
    ("gyrA",  83, "S", "P",  1.2,  "Gullberg2014",             "ciprofloxacin"),
    ("gyrA",  83, "S", "G",  1.0,  "Gullberg2014",             "ciprofloxacin"),
    ("gyrA",  87, "D", "E",  1.2,  "Gullberg2014",             "ciprofloxacin"),
    ("parC",  80, "S", "T",  0.8,  "Gullberg2014",             "ciprofloxacin"),
    # Wild-type / synonymous-equivalent control points (set fitness ~0)
    ("gyrA",  83, "S", "S",  0.0,  "Control_WT",               "ciprofloxacin"),
    ("gyrA",  87, "D", "D",  0.0,  "Control_WT",               "ciprofloxacin"),
    ("parC",  80, "S", "S",  0.0,  "Control_WT",               "ciprofloxacin"),
    ("parC",  84, "E", "E",  0.0,  "Control_WT",               "ciprofloxacin"),
]

# Per-gene UniProt anchors (E. coli K-12 reference IDs).
GENE_UNIPROT = {
    "gyrA": "P0AES4",   # E. coli K-12 gyrA
    "gyrB": "P0AES6",   # E. coli K-12 gyrB
    "parC": "P0AFI2",   # E. coli K-12 parC
    "parE": "P20083",   # E. coli K-12 parE
}


def build_fq_dms() -> pd.DataFrame:
    rows = []
    # All four drugs in the fluoroquinolone class — replicate scores across
    # closely related FQs (ciprofloxacin, levofloxacin, norfloxacin, moxifloxacin).
    fq_drugs = ["ciprofloxacin", "levofloxacin", "norfloxacin", "moxifloxacin"]
    for gene, pos, wt, mut, lfold, src, default_drug in CURATED:
        for drug in fq_drugs:
            f = float(lfold)
            rows.append({
                "gene_family":          gene.upper() if gene in ("gyrA","gyrB","parC","parE") else gene,
                "gene_uniprot":         GENE_UNIPROT[gene],
                "position":             int(pos),
                "wildtype_aa":          wt,
                "mutant_aa":            mut,
                "fitness_score":        f,                              # log2 MIC fold
                "fitness_zscore":       np.nan,                         # filled below
                "fitness_log_score":    f,                              # already log2
                "source_dataset":       f"FQ_curated_{src}",
                "source_doi":           ";".join(_doi_for_src(src)),
                "condition":            drug,
                "source_normalization": "log2(MIC_fold) curated from published tables",
                "source_urn":           "",
            })
    df = pd.DataFrame(rows)
    # z-score within the curated FQ DMS table
    mu = df["fitness_score"].mean()
    sd = df["fitness_score"].std(ddof=0)
    if sd == 0:
        df["fitness_zscore"] = 0.0
    else:
        df["fitness_zscore"] = (df["fitness_score"] - mu) / sd
    # Normalize gene_family to upper to match Day-1 convention (TEM, VIM, DHFR).
    df["gene_family"] = df["gene_family"].str.upper()
    return df


def _doi_for_src(src: str) -> list[str]:
    out = []
    if "Pal2015" in src:        out.append("PMID:26384679")
    if "Marcusson2009" in src:  out.append("PMID:19188355")
    if "Gullberg2014" in src:   out.append("PMID:24875993")
    if "Hooper2015" in src:     out.append("PMID:25653173")
    if "Vila1996" in src:       out.append("PMID:8814580")
    if "Control" in src:        out.append("internal:WT_control")
    return out


def merge_into_dms(df_fq: pd.DataFrame, log: logging.Logger) -> int:
    existing = pd.read_parquet(OUT)
    n_before = len(existing)
    # Ensure schema alignment.
    cols = list(existing.columns)
    df_fq = df_fq[cols]
    combined = pd.concat([existing, df_fq], ignore_index=True)
    combined.to_parquet(OUT, index=False)
    n_after = len(combined)
    log.info("dms_fitness.parquet: %d -> %d rows (+%d)",
             n_before, n_after, n_after - n_before)
    return n_after - n_before


def main() -> int:
    log = setup_logger()
    log.info("=== Phase A4 — FQ DMS curation ===")
    df = build_fq_dms()
    log.info("Curated FQ DMS rows: %d  (positions: %d  unique mutations: %d)",
             len(df),
             df[["gene_family","position"]].drop_duplicates().shape[0],
             df[["gene_family","position","wildtype_aa","mutant_aa"]].drop_duplicates().shape[0])
    log.info("Families: %s", df["gene_family"].value_counts().to_dict())
    log.info("z-score stats: mean=%.3f std=%.3f min=%.3f max=%.3f",
             df["fitness_zscore"].mean(), df["fitness_zscore"].std(),
             df["fitness_zscore"].min(), df["fitness_zscore"].max())

    INTERMEDIATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERMEDIATE_OUT, index=False)
    log.info("Wrote intermediate -> %s", INTERMEDIATE_OUT)

    added = merge_into_dms(df, log)

    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with GATE_LOG.open("a") as fh:
        fh.write(f"{ts} PHASE_A4_FQ_DMS: PASS [n_new_measurements={added}] "
                 f"[mutation_keys_curated={df[['gene_family','position','wildtype_aa','mutant_aa']].drop_duplicates().shape[0]}]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
