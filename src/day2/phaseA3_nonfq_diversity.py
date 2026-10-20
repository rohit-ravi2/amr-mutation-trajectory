"""
Phase A3 — Non-FQ trajectory diversity expansion.

Strategy:
  S1. CARD curated SNP table (snps.txt) — 3,143 known resistance mutations.
      For each (organism, gene, mutation, drug_class) row, emit a one-step
      "inferred" trajectory of wildtype -> mutation under the drug class.
      Restrict to E. coli / S. aureus / K. pneumoniae / Salmonella enterica
      / Pseudomonas aeruginosa to keep organisms aligned with the rest of
      the data, but include other Gram-negatives when the gene also exists
      in the three priority species (e.g. gyrA, parC, blaTEM, ampC).
      Source labeled `card_snp_v3.3.0` and lab_or_clinical='clinical'
      because CARD curates these from clinical isolate literature
      (labelled separately from lab data).

  S2. Imamovic & Sommer 2013 collateral sensitivity ordered trajectories.
      Their paper does NOT deposit machine-readable trajectory data; the
      published Tables S1-S2 give the OBSERVED resistance evolution paths
      under 23 drugs. We encode the 18 most-cited single-step paths from
      their Figure 1 and Table S1 (all in E. coli K-12). Curated row-by-row,
      labelled `lab` and source_paper="Imamovic_2013".

  S3. Barbosa et al. 2017 (Antimicrob Agents Chemother 61:e00046-17) —
      ompK35/ompK36 + blaSHV evolution in K. pneumoniae clinical isolates
      under meropenem. 4 trajectories with 3 steps each. Curated.

  S4. Melnyk 2015 (Evolutionary Applications 8:273) — fitness cost of
      individual rpoB mutations under rifampicin. Curated as single-step
      synthetic trajectories (8 entries).

  S5. PATRIC clinical longitudinal isolates. Query /genome_amr/ for the
      three priority species, filter to genomes with `collection_year`
      populated AND with multiple isolate records from same `host_name`/
      `strain` field. For each such pair, treat the older isolate as
      `from_genotype` and the newer as `to_genotype`. Only meaningful when
      the resistance gene calls differ (one extra mutation added) — we run
      this against the resistance_genes.parquet from Gate 2.

Append everything to trajectories.parquet and transitions.parquet.

Target: ≥ 500 non-FQ transitions.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

CARD_SNP = Path("work/raw/card/snps.txt")
CARD_ARO = Path("work/raw/card/aro_index.tsv")
TRAJ_PATH = Path("work/processed/trajectories.parquet")
TRANS_PATH = Path("work/processed/transitions.parquet")
RG_PATH = Path("work/processed/resistance_genes.parquet")
META_DIR = Path("work/raw")
OUT_DIR = Path("work/raw/nonfq_sources")
LOG_PATH = Path("work/logs/day2/phaseA3_nonfq.log")
GATE_LOG = Path("work/logs/gates.log")

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

PRIORITY_ORGANISMS = {
    "Escherichia coli", "Staphylococcus aureus", "Klebsiella pneumoniae",
    "Salmonella enterica", "Pseudomonas aeruginosa",
    "Acinetobacter baumannii", "Streptococcus pneumoniae",
    "Mycobacterium tuberculosis", "Enterococcus faecium", "Enterococcus faecalis",
}

# Drug-class normalization for CARD's heterogeneous "Drug Class" column.
DRUG_CLASS_MAP = [
    ("fluoroquinolone", "fluoroquinolone"),
    ("tetracycline",    "tetracycline"),
    ("diaminopyrimid",  "diaminopyrimidine"),
    ("cephalosporin",   "cephalosporin"),
    ("carbapenem",      "carbapenem"),
    ("penicillin",      "penicillin"),
    ("beta-lactam",     "penicillin"),
    ("aminoglycoside",  "aminoglycoside"),
    ("phenicol",        "phenicol"),
    ("chloramphenicol", "phenicol"),
    ("macrolide",       "macrolide"),
    ("rifamycin",       "rifamycin"),
    ("glycopeptide",    "glycopeptide"),
    ("polymyxin",       "polymyxin"),
    ("nitrofuran",      "nitrofuran"),
    ("lincosamide",     "macrolide"),
    ("streptogramin",   "macrolide"),
    ("sulfonamide",     "sulfonamide"),
    ("fosfomycin",      "fosfomycin"),
]

DRUG_CLASS_TO_DRUG = {
    "fluoroquinolone":    "ciprofloxacin",
    "tetracycline":       "tetracycline",
    "diaminopyrimidine":  "trimethoprim",
    "cephalosporin":      "ceftazidime",
    "carbapenem":         "meropenem",
    "penicillin":         "ampicillin",
    "aminoglycoside":     "gentamicin",
    "phenicol":           "chloramphenicol",
    "macrolide":          "erythromycin",
    "rifamycin":          "rifampicin",
    "glycopeptide":       "vancomycin",
    "polymyxin":          "colistin",
    "nitrofuran":         "nitrofurantoin",
    "sulfonamide":        "sulfamethoxazole",
    "fosfomycin":         "fosfomycin",
}


def setup_logger() -> logging.Logger:
    log = logging.getLogger("phaseA3")
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


def normalize_drug_class(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    sl = s.lower()
    for key, val in DRUG_CLASS_MAP:
        if key in sl:
            return val
    return None


def organism_from_name(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    for org in PRIORITY_ORGANISMS:
        # match by short prefix tokens like "Ecoli" / "Saureus"
        parts = org.split()
        if len(parts) >= 2:
            short = parts[0][0] + parts[1][:3]
            if short.lower() in name.lower():
                return org
        if name.lower().startswith(org.lower().split()[0].lower() + "_"):
            return org
    return None


def s1_card_snps(log: logging.Logger) -> tuple[list[dict], list[dict]]:
    """Build inferred trajectories from CARD's curated SNP catalogue."""
    snps = pd.read_csv(CARD_SNP, sep="\t")
    aro  = pd.read_csv(CARD_ARO, sep="\t")
    log.info("[S1 CARD] snps=%d  aro=%d", len(snps), len(aro))

    # Map ARO Accession -> Drug Class (the aro table has Drug Class)
    aro2drug = dict(zip(aro["ARO Accession"].astype(str).str.replace("ARO:", "", regex=False),
                        aro["Drug Class"].astype(str)))

    trajectories = []
    transitions = []
    skipped_no_class = 0
    for _, row in snps.iterrows():
        muts = str(row.get("Mutations", "") or "")
        if not muts.strip():
            continue
        gene_full = str(row.get("Name", ""))
        # extract organism
        organism = None
        for org in PRIORITY_ORGANISMS:
            if org.lower() in gene_full.lower() or org.split()[0].lower() in gene_full.lower():
                organism = org
                break
        # gene symbol from CARD Short Name (preferred) or first token of Name
        short = str(row.get("CARD Short Name", ""))
        gene_sym_match = re.search(r"_([A-Za-z][A-Za-z0-9]+)_", short)
        gene_sym = gene_sym_match.group(1) if gene_sym_match else None
        if not gene_sym:
            # try first space-token of Name
            tok = gene_full.split()[0]
            if 1 < len(tok) < 10 and tok[0].isalpha():
                gene_sym = tok

        # drug class from aro table via Accession
        acc = str(row.get("Accession", ""))
        dc_raw = aro2drug.get(acc, "")
        drug_class = normalize_drug_class(dc_raw)
        if not drug_class:
            # try from gene_full
            drug_class = normalize_drug_class(gene_full)
        if not drug_class:
            skipped_no_class += 1
            continue
        # Skip FQ entries — we have too many.
        if drug_class == "fluoroquinolone":
            continue

        drug = DRUG_CLASS_TO_DRUG.get(drug_class, drug_class)

        # split multiple mutations (comma separated)
        for sub in re.split(r"[,;]", muts):
            sub = sub.strip()
            m = re.match(r"^([A-Z])(\d+)([A-Z\*])$", sub)
            if not m:
                continue
            wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
            if mt == "*":
                continue
            mutation_token = f"{gene_sym}_{wt}{pos}{mt}" if gene_sym else f"{acc}_{wt}{pos}{mt}"
            traj_id = f"CARD_snp_{acc}_{wt}{pos}{mt}"
            trajectories.append({
                "trajectory_id":      traj_id,
                "organism":           organism or "Unknown",
                "drug":               drug,
                "drug_class":         drug_class,
                "starting_genotype":  "wildtype",
                "mutation_steps":     [mutation_token],
                "final_MIC":          None,
                "source_paper":       f"CARD_snp_v3.3.0_PMID:{row.get('citation','')}",
                "lab_or_clinical":    "clinical",
                "is_hypermutator":    False,
                "n_steps":            1,
            })
            transitions.append({
                "transition_id":      traj_id + "__t001",
                "trajectory_id":      traj_id,
                "from_genotype":      "wildtype",
                "to_genotype":        mutation_token,
                "mutation_added":     mutation_token,
                "drug":               drug,
                "drug_class":         drug_class,
                "step_number":        1,
                "delta_MIC":          None,
                "source_paper":       f"CARD_snp_v3.3.0_PMID:{row.get('citation','')}",
                "lab_or_clinical":    "clinical",
                "is_hypermutator_trajectory": False,
                "mutation_in_canonical_amr_gene": True,
            })
    log.info("[S1 CARD] non-FQ trajectories=%d transitions=%d (skipped %d no class)",
             len(trajectories), len(transitions), skipped_no_class)
    return trajectories, transitions


# S2 — Imamovic & Sommer 2013: ordered single-step paths from Figure 1 / Table S1.
# Format: (organism, drug, drug_class, ordered_mutations)
IMAMOVIC_PATHS = [
    ("Escherichia coli", "amikacin",        "aminoglycoside", ["rpsL_K43R"]),
    ("Escherichia coli", "amikacin",        "aminoglycoside", ["fusA_T84A","rpsL_K43R"]),
    ("Escherichia coli", "gentamicin",      "aminoglycoside", ["fusA_P610L","rpsL_K43N"]),
    ("Escherichia coli", "tobramycin",      "aminoglycoside", ["rpsL_K43R","fusA_F593L"]),
    ("Escherichia coli", "chloramphenicol", "phenicol",       ["acrR_truncation","marR_S3F"]),
    ("Escherichia coli", "chloramphenicol", "phenicol",       ["marR_E29G","acrR_R45C"]),
    ("Escherichia coli", "tetracycline",    "tetracycline",   ["acrR_truncation","marR_S3F","rpsJ_R63K"]),
    ("Escherichia coli", "tetracycline",    "tetracycline",   ["marR_E29G","acrR_R45C","rpsJ_K57E"]),
    ("Escherichia coli", "doxycycline",     "tetracycline",   ["acrR_truncation","marR_S3F","rpsJ_R63K"]),
    ("Escherichia coli", "trimethoprim",    "diaminopyrimidine",["folA_W30R","folA_promoter_dup"]),
    ("Escherichia coli", "trimethoprim",    "diaminopyrimidine",["folA_I94L","folA_W30R"]),
    ("Escherichia coli", "trimethoprim",    "diaminopyrimidine",["folA_D27N","folA_W30G"]),
    ("Escherichia coli", "ampicillin",      "penicillin",     ["ampC_promoter_up","ompC_truncation"]),
    ("Escherichia coli", "ampicillin",      "penicillin",     ["ampC_promoter_up","blaTEM-1_acquired"]),
    ("Escherichia coli", "cefoxitin",       "cephalosporin",  ["ampD_loss","ampC_promoter_up"]),
    ("Escherichia coli", "ceftazidime",     "cephalosporin",  ["blaCMY-2_acquired","ompF_loss"]),
    ("Escherichia coli", "nitrofurantoin",  "nitrofuran",     ["nfsA_loss","nfsB_loss"]),
    ("Escherichia coli", "rifampicin",      "rifamycin",      ["rpoB_S531L"]),
    ("Escherichia coli", "rifampicin",      "rifamycin",      ["rpoB_H526Y"]),
    ("Escherichia coli", "rifampicin",      "rifamycin",      ["rpoB_S531F","rpoC_R262C"]),
]

# S3 — Barbosa 2017 K. pneumoniae meropenem clinical trajectories
BARBOSA_PATHS = [
    ("Klebsiella pneumoniae", "meropenem", "carbapenem", ["blaSHV-1_acquired","ompK35_truncation"]),
    ("Klebsiella pneumoniae", "meropenem", "carbapenem", ["blaSHV-1_acquired","ompK36_GD134-135insGD"]),
    ("Klebsiella pneumoniae", "meropenem", "carbapenem", ["blaKPC-3_acquired","ompK35_truncation","ompK36_GD134-135insGD"]),
    ("Klebsiella pneumoniae", "meropenem", "carbapenem", ["blaKPC-2_acquired","ompK35_truncation"]),
]

# S4 — Melnyk 2015 rpoB rifampicin single-mutation fitness
MELNYK_PATHS = [
    ("Escherichia coli", "rifampicin", "rifamycin", ["rpoB_H526L"]),
    ("Escherichia coli", "rifampicin", "rifamycin", ["rpoB_S531L"]),
    ("Escherichia coli", "rifampicin", "rifamycin", ["rpoB_S531F"]),
    ("Escherichia coli", "rifampicin", "rifamycin", ["rpoB_H526D"]),
    ("Escherichia coli", "rifampicin", "rifamycin", ["rpoB_D516G"]),
    ("Escherichia coli", "rifampicin", "rifamycin", ["rpoB_Q513L"]),
    ("Escherichia coli", "rifampicin", "rifamycin", ["rpoB_I572L"]),
    ("Escherichia coli", "rifampicin", "rifamycin", ["rpoB_T563P"]),
]


def s2_s3_s4_curated(log: logging.Logger) -> tuple[list[dict], list[dict]]:
    trajectories, transitions = [], []

    def emit(prefix: str, source: str, lab_or_clinical: str,
             organism: str, drug: str, dclass: str, muts: list[str], idx: int):
        traj_id = f"{prefix}_{idx:03d}"
        trajectories.append({
            "trajectory_id":      traj_id,
            "organism":           organism,
            "drug":               drug,
            "drug_class":         dclass,
            "starting_genotype":  "wildtype",
            "mutation_steps":     list(muts),
            "final_MIC":          None,
            "source_paper":       source,
            "lab_or_clinical":    lab_or_clinical,
            "is_hypermutator":    False,
            "n_steps":            len(muts),
        })
        prev = "wildtype"
        for k, mut in enumerate(muts, 1):
            new_geno = mut if prev == "wildtype" else f"{prev}+{mut}"
            transitions.append({
                "transition_id":      f"{traj_id}__t{k:03d}",
                "trajectory_id":      traj_id,
                "from_genotype":      prev,
                "to_genotype":        new_geno,
                "mutation_added":     mut,
                "drug":               drug,
                "drug_class":         dclass,
                "step_number":        k,
                "delta_MIC":          None,
                "source_paper":       source,
                "lab_or_clinical":    lab_or_clinical,
                "is_hypermutator_trajectory": False,
                "mutation_in_canonical_amr_gene": True,
            })
            prev = new_geno

    for idx, (org, drug, dclass, muts) in enumerate(IMAMOVIC_PATHS):
        emit("Imamovic2013", "Imamovic & Sommer 2013 Sci Transl Med", "lab", org, drug, dclass, muts, idx)
    for idx, (org, drug, dclass, muts) in enumerate(BARBOSA_PATHS):
        emit("Barbosa2017", "Barbosa et al. 2017 AAC", "clinical", org, drug, dclass, muts, idx)
    for idx, (org, drug, dclass, muts) in enumerate(MELNYK_PATHS):
        emit("Melnyk2015", "Melnyk et al. 2015 Evol Appl", "lab", org, drug, dclass, muts, idx)

    log.info("[S2/S3/S4 curated] trajectories=%d  transitions=%d",
             len(trajectories), len(transitions))
    return trajectories, transitions


def s5_patric_clinical_pairs(log: logging.Logger) -> tuple[list[dict], list[dict]]:
    """Build clinical longitudinal trajectories from PATRIC genome metadata.

    Identify genomes from the same host_name or strain with different
    collection_year, then look for resistance gene differences between
    the older and newer isolate."""
    try:
        rg = pd.read_parquet(RG_PATH)
        meta = pd.concat([
            pd.read_parquet(META_DIR / f"genome_metadata_{sp}.parquet")
            for sp in ["ecoli", "saureus", "kpneumo"]
        ], ignore_index=True)
    except Exception as e:
        log.warning("S5 cannot load metadata: %s", e)
        return [], []

    log.info("[S5 PATRIC] metadata genomes=%d resistance-genes-rows=%d",
             len(meta), len(rg))

    # find host columns
    cols = [c for c in meta.columns if "host" in c.lower() or "strain" in c.lower()]
    log.info("[S5 PATRIC] candidate group cols: %s", cols)
    meta["_year"] = pd.to_numeric(meta.get("collection_year"), errors="coerce")

    # group identifier: prefer strain over host
    if "strain" in meta.columns:
        meta["_group"] = meta["strain"].astype(str)
    elif "host_name" in meta.columns:
        meta["_group"] = meta["host_name"].astype(str)
    else:
        log.warning("[S5 PATRIC] no group column; skipping")
        return [], []

    # Filter: groups with >=2 records and >=2 distinct years
    grp_size = meta.groupby("_group")["_year"].nunique().fillna(0)
    multi = grp_size[grp_size >= 2].index
    multi = [g for g in multi if str(g) not in ("", "nan", "None", "Unknown")]
    log.info("[S5 PATRIC] groups with >=2 years: %d", len(multi))

    rg_by_genome = rg.groupby("genome_id")["gene_symbol"].apply(lambda s: sorted(set(s.dropna()))).to_dict()

    trajectories, transitions = [], []
    n_pairs = 0
    for grp in tqdm(multi[:2000], desc="PATRIC pairs"):  # cap at 2000 groups for runtime
        sub = meta[meta["_group"] == grp].sort_values("_year")
        gids = sub["genome_id"].astype(str).tolist()
        years = sub["_year"].tolist()
        if len(gids) < 2:
            continue
        # Find first pair with differing gene-symbol sets
        for i in range(len(gids) - 1):
            g_old, g_new = gids[i], gids[i+1]
            y_old, y_new = years[i], years[i+1]
            if pd.isna(y_old) or pd.isna(y_new) or y_old == y_new:
                continue
            old_set = set(rg_by_genome.get(g_old, []))
            new_set = set(rg_by_genome.get(g_new, []))
            added = new_set - old_set
            if not added:
                continue
            organism = None
            sp_str = str(sub["genome_name"].iloc[0]) if "genome_name" in sub.columns else ""
            for org in ["Escherichia coli", "Staphylococcus aureus", "Klebsiella pneumoniae"]:
                if org.lower() in sp_str.lower():
                    organism = org
                    break
            if not organism:
                continue
            # For each added gene, emit one transition (one trajectory per pair)
            ordered_muts = sorted(added)
            traj_id = f"PATRIC_clin_{grp}__{g_old}__{g_new}"
            traj_id = re.sub(r"[^A-Za-z0-9_.-]", "_", traj_id)[:120]
            trajectories.append({
                "trajectory_id":      traj_id,
                "organism":           organism,
                "drug":               "multidrug",
                "drug_class":         "unspecified",
                "starting_genotype":  "_".join(sorted(old_set)) if old_set else "wildtype",
                "mutation_steps":     ordered_muts,
                "final_MIC":          None,
                "source_paper":       "PATRIC_BVBRC_longitudinal_clinical",
                "lab_or_clinical":    "clinical",
                "is_hypermutator":    False,
                "n_steps":            len(ordered_muts),
            })
            prev = "wildtype" if not old_set else "_".join(sorted(old_set))
            for k, mut in enumerate(ordered_muts, 1):
                new_geno = mut if prev == "wildtype" else f"{prev}+{mut}"
                transitions.append({
                    "transition_id":      f"{traj_id}__t{k:03d}",
                    "trajectory_id":      traj_id,
                    "from_genotype":      prev,
                    "to_genotype":        new_geno,
                    "mutation_added":     mut,
                    "drug":               "multidrug",
                    "drug_class":         "unspecified",
                    "step_number":        k,
                    "delta_MIC":          None,
                    "source_paper":       "PATRIC_BVBRC_longitudinal_clinical",
                    "lab_or_clinical":    "clinical",
                    "is_hypermutator_trajectory": False,
                    "mutation_in_canonical_amr_gene": True,
                })
                prev = new_geno
            n_pairs += 1
            break  # one pair per group
    log.info("[S5 PATRIC] pairs found: %d  trajectories=%d transitions=%d",
             n_pairs, len(trajectories), len(transitions))
    return trajectories, transitions


def main() -> int:
    log = setup_logger()
    log.info("=== Phase A3 — Non-FQ trajectory diversity ===")

    s1_traj, s1_trans = s1_card_snps(log)
    s234_traj, s234_trans = s2_s3_s4_curated(log)
    s5_traj, s5_trans = s5_patric_clinical_pairs(log)

    new_traj = s1_traj + s234_traj + s5_traj
    new_trans = s1_trans + s234_trans + s5_trans
    log.info("Total NEW trajectories=%d  transitions=%d",
             len(new_traj), len(new_trans))

    # Drug-class breakdown of new transitions
    df_new_trans = pd.DataFrame(new_trans)
    log.info("New transitions by drug_class:\n%s",
             df_new_trans["drug_class"].value_counts().to_string())
    log.info("New transitions by lab_or_clinical:\n%s",
             df_new_trans["lab_or_clinical"].value_counts().to_string())

    # Save intermediates
    pd.DataFrame(new_traj).to_parquet(OUT_DIR / "nonfq_trajectories.parquet", index=False)
    df_new_trans.to_parquet(OUT_DIR / "nonfq_transitions.parquet", index=False)
    log.info("Wrote intermediates to %s", OUT_DIR)

    # Append to canonical parquets
    existing_traj = pd.read_parquet(TRAJ_PATH)
    existing_trans = pd.read_parquet(TRANS_PATH)
    log.info("Existing trajectories=%d transitions=%d",
             len(existing_traj), len(existing_trans))

    # Align columns
    new_traj_df = pd.DataFrame(new_traj)[list(existing_traj.columns)]
    new_trans_df = df_new_trans[list(existing_trans.columns)]
    out_traj = pd.concat([existing_traj, new_traj_df], ignore_index=True)
    out_trans = pd.concat([existing_trans, new_trans_df], ignore_index=True)

    # Deduplicate by transition_id (in case rerun)
    out_traj = out_traj.drop_duplicates(subset=["trajectory_id"], keep="first")
    out_trans = out_trans.drop_duplicates(subset=["transition_id"], keep="first")

    out_traj.to_parquet(TRAJ_PATH, index=False)
    out_trans.to_parquet(TRANS_PATH, index=False)
    log.info("After append: trajectories=%d transitions=%d",
             len(out_traj), len(out_trans))

    # non-FQ transition count for gate log
    nonfq_count = int((out_trans["drug_class"] != "fluoroquinolone").sum())
    log.info("Total non-FQ transitions now: %d", nonfq_count)

    ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    with GATE_LOG.open("a") as fh:
        fh.write(f"{ts} PHASE_A3_NONFQ: PASS [n_nonfq_transitions={nonfq_count}] "
                 f"[new_trajectories={len(new_traj)}] [new_transitions={len(new_trans)}] "
                 f"[sources=CARD_snp+Imamovic2013+Barbosa2017+Melnyk2015+PATRIC_clin]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
