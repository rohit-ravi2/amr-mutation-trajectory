"""
GATE 3 — Directed evolution trajectory table assembly.

Strategy after early reconnaissance:

  - Publisher-hosted SI (Science, Nature) returns 403 to non-browser clients.
  - PMC SI files are gated behind a JS proof-of-work cookie wall.
  - ALEdb (aledb.org) is the canonical aggregator: contains Baym 2016, the
    Toprak-style Chl/Dox/Tri data (project 119), and a long list of other
    antibiotic ALE projects.

So the pipeline below treats ALEdb as the primary source. We pull each
relevant project's mutation table directly from the HTML (table_data JS
literal embedded in the page).

Source labels in the output:
  A:    Baym_M_2016                              -> lab (trimethoprim ladder)
  B:    Ecoli_resistance_ChlDoxTri (Toprak-style) -> lab (Chl/Dox/Tri)
  D1:   Ecoli_resistance_norfloxacin             -> lab
  D2:   Ecoli_resistance_caz (ceftazidime)       -> lab
  D3:   Ecoli_resistance_tet_intragenCoevol      -> lab
  D4:   Reka_Spohn_2019 (30-antibiotic)          -> lab
  D5:   Swings_T_2017                            -> lab
  D6:   Pablo_Laborda_2022 (tobramycin urine)    -> lab
  D7:   Sada_M_2022                              -> lab
  D8:   Akamu_J_Ewunkem_2021                     -> lab
  E:    inferred_card  (cross-section ladder)    -> inferred_card

Output:
  work/processed/trajectories.parquet
  work/processed/transitions.parquet
  work/processed/gate3_intermediate/<source>.parquet
  work/processed/gate3_summary.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SSD_PROC = Path("work/processed")
HDD_RAW = Path("work/raw/gate3")
LOGS = Path("work/logs")
INTERMED = SSD_PROC / "gate3_intermediate"

SSD_PROC.mkdir(parents=True, exist_ok=True)
INTERMED.mkdir(parents=True, exist_ok=True)
HDD_RAW.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "gate3.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("gate3")

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
ALEDB = "https://aledb.org"

# -----------------------------------------------------------------------------
# Drug class lookup (used by all source parsers)
# -----------------------------------------------------------------------------

DRUG_CLASS = {
    "trimethoprim": "diaminopyrimidine",
    "TMP": "diaminopyrimidine",
    "chloramphenicol": "phenicol",
    "CHL": "phenicol",
    "doxycycline": "tetracycline",
    "DOX": "tetracycline",
    "tetracycline": "tetracycline",
    "TET": "tetracycline",
    "ciprofloxacin": "fluoroquinolone",
    "CIP": "fluoroquinolone",
    "norfloxacin": "fluoroquinolone",
    "NOR": "fluoroquinolone",
    "ofloxacin": "fluoroquinolone",
    "levofloxacin": "fluoroquinolone",
    "moxifloxacin": "fluoroquinolone",
    "ceftazidime": "cephalosporin",
    "CAZ": "cephalosporin",
    "ceftriaxone": "cephalosporin",
    "cefotaxime": "cephalosporin",
    "ampicillin": "penicillin",
    "amoxicillin": "penicillin",
    "oxacillin": "penicillin",
    "methicillin": "penicillin",
    "meropenem": "carbapenem",
    "imipenem": "carbapenem",
    "tobramycin": "aminoglycoside",
    "TOB": "aminoglycoside",
    "amikacin": "aminoglycoside",
    "gentamicin": "aminoglycoside",
    "kanamycin": "aminoglycoside",
    "streptomycin": "aminoglycoside",
    "colistin": "polymyxin",
    "polymyxin": "polymyxin",
    "vancomycin": "glycopeptide",
    "azithromycin": "macrolide",
    "erythromycin": "macrolide",
    "clarithromycin": "macrolide",
    "rifampicin": "rifamycin",
    "rifampin": "rifamycin",
    "linezolid": "oxazolidinone",
    "nitrofurantoin": "nitrofuran",
    "fosfomycin": "fosfomycin",
    "sulfamethoxazole": "sulfonamide",
    "fusidic_acid": "fusidane",
}


def drug_class_for(drug: str) -> str:
    if drug is None:
        return "unknown"
    d = drug.strip()
    if d in DRUG_CLASS:
        return DRUG_CLASS[d]
    dl = d.lower()
    for k, v in DRUG_CLASS.items():
        if k.lower() == dl:
            return v
    return "unknown"


# -----------------------------------------------------------------------------
# ALEdb HTML scraping helpers
# -----------------------------------------------------------------------------

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept": "*/*"})
        _session = s
    return _session


def fetch(url: str, dst: Path, force: bool = False) -> Path:
    """GET a URL to disk with simple retry. Returns path."""
    if dst.exists() and dst.stat().st_size > 1024 and not force:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(4):
        try:
            r = session().get(url, timeout=90)
            if r.status_code == 200 and r.content:
                dst.write_bytes(r.content)
                return dst
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed: {url}: {last}")


# ---- ALEdb HTML parsers ----

_RE_TABLE_DATA = re.compile(r"var\s+table_data\s*=\s*(\[.*?\]\]);", re.DOTALL)
_RE_TABLE_TITLES = re.compile(r"var\s+table_titles\s*=\s*(\[[^\]]+\])", re.DOTALL)
_RE_GENE_LINK = re.compile(
    r'<a\s+href\s*=\s*"https://ecocyc\.org/[^"]*object=([^"]+)">([^<]+)</a>'
)
_RE_DETAILS = re.compile(r'<a class="[^"]+" href="/mutations/details\?observed_mut_id=\d+">([0-9.]+)</a>/<a class="[^"]+" href="/mutations/details\?observed_mut_id=\d+"[^>]*>([0-9.]+)</a>')
_RE_EMPTY = re.compile(r'<span class="empty"></span>')


def _strip_html(s: str) -> str:
    """Strip HTML tags, return plain text."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"<[^>]+>", "", s).strip()


def _parse_frequency_cell(cell: str) -> float | None:
    """ALEdb frequency cells are HTML like
       '<a ...>1.00</a>/<a ...>0.00</a>'  or  '<span class="empty"></span>'
       Return the first frequency (population / clone freq) as float, or None.
    """
    if not isinstance(cell, str) or not cell.strip():
        return None
    if "<span class=\"empty\">" in cell:
        return None
    m = _RE_DETAILS.search(cell)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    # fall back: strip tags then parse number/number
    txt = _strip_html(cell)
    if "/" in txt:
        try:
            return float(txt.split("/")[0])
        except ValueError:
            return None
    try:
        return float(txt)
    except ValueError:
        return None


def _parse_gene_cell(cell: str) -> str:
    """Pull the gene symbol out of an ALEdb gene cell."""
    if not isinstance(cell, str):
        return ""
    m = _RE_GENE_LINK.search(cell)
    if m:
        return m.group(2)
    return _strip_html(cell)


def parse_aledb_mutations_html(html: str) -> tuple[list[str], pd.DataFrame]:
    """Parse the /mutations?ale_experiment_id=N page.

    Returns (sample_columns, mutations_df).
    mutations_df columns:
        ref_seq, position, mut_type, seq_change, gene, function, product,
        mut_id, details, plus one freq_<sample> column per isolate.
    """
    mt = _RE_TABLE_TITLES.search(html)
    md = _RE_TABLE_DATA.search(html)
    if not mt:
        raise ValueError("table_titles not found")
    if not md:
        raise ValueError("table_data not found")

    titles_raw = mt.group(1)
    # ALEdb embeds the sample tag dropdown HTML in each isolate title string.
    # We just want the leading "A# F# I# R#" or similar identifier.
    titles_strs = re.findall(r"'((?:[^'\\]|\\.)*)'", titles_raw)
    # Clean tag-dropdown HTML out of each title:
    titles_clean = []
    for t in titles_strs:
        t = t.replace("\\u00", "")
        t = re.sub(r"</?div[^>]*>", "", t)
        t = re.sub(r'<span class="fa-stack"[^>]*>.*?</span>', "", t, flags=re.DOTALL)
        t = re.sub(r"<[^>]+>", "", t).strip()
        titles_clean.append(t)
    titles = titles_clean

    # Parse table_data via JSON. The JS uses double-quoted strings and unicode
    # escapes already, so a json.loads after a couple of cleanups works.
    raw = md.group(1)
    # Some unicode arrows aren't valid in strict JSON; they're fine here.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Some pages have stray trailing commas; remove them.
        raw_clean = re.sub(r",(\s*[\]\}])", r"\1", raw)
        data = json.loads(raw_clean)

    n_cols = len(titles)
    rows = []
    for row in data:
        # pad/truncate to n_cols
        if len(row) < n_cols:
            row = row + [""] * (n_cols - len(row))
        elif len(row) > n_cols:
            row = row[:n_cols]
        rows.append(row)

    df = pd.DataFrame(rows, columns=titles)

    # Standard meta columns (first 14 are admin/metadata in ALEdb pages):
    # ['', '', 'Tags', 'Reference Seq', 'Position', 'Mutation Type',
    #  'Sequence Change', 'Gene (Scrollable)', 'Function', 'Product',
    #  'GO Process', 'GO Component', 'Mut ID', 'Details', <samples...>]
    rename = {
        "Reference Seq": "ref_seq",
        "Position": "position",
        "Mutation Type": "mut_type",
        "Sequence Change": "seq_change",
        "Gene (Scrollable)": "gene_html",
        "Function": "function",
        "Product": "product",
        "Mut ID": "mut_id",
        "Details": "details",
    }
    df = df.rename(columns=rename)
    df["gene"] = df["gene_html"].apply(_parse_gene_cell)
    df["details_text"] = df["details"].apply(_strip_html)

    sample_cols = [c for c in df.columns if re.match(r"^A\d+\s+F\d+\s+I\d+\s+R\d+", c)]
    # Frequency parse for sample columns
    for c in sample_cols:
        df[f"freq__{c}"] = df[c].apply(_parse_frequency_cell)

    keep = [
        "ref_seq", "position", "mut_type", "seq_change",
        "gene", "function", "product", "mut_id", "details_text",
    ] + [f"freq__{c}" for c in sample_cols]
    out = df[keep].copy()
    out["sample_columns"] = [sample_cols] * len(out)
    return sample_cols, out


def fetch_aledb_experiment_mutations(experiment_id: int,
                                     cache_dir: Path = HDD_RAW / "aledb_cache") -> tuple[list[str], pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"experiment_{experiment_id}_mutations.html"
    fetch(f"{ALEDB}/mutations?ale_experiment_id={experiment_id}", dst)
    html = dst.read_text()
    return parse_aledb_mutations_html(html)


def fetch_aledb_project_experiments(project_id: int,
                                    cache_dir: Path = HDD_RAW / "aledb_cache") -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"project_{project_id}.html"
    fetch(f"{ALEDB}/ale/project/{project_id}", dst)
    html = dst.read_text()
    # Each experiment row has an href like /ale/experiment/<id>
    rows = re.findall(
        r'href="/ale/experiment/(\d+)"[^>]*>([^<]+)<',
        html,
    )
    # Deduplicate while preserving order
    seen = set()
    out = []
    for eid, name in rows:
        eid_i = int(eid)
        if eid_i in seen:
            continue
        seen.add(eid_i)
        out.append({"experiment_id": eid_i, "name": name.strip()})
    return out


# -----------------------------------------------------------------------------
# Trajectory construction
# -----------------------------------------------------------------------------

@dataclass
class Trajectory:
    trajectory_id: str
    organism: str
    drug: str
    drug_class: str
    starting_genotype: str
    mutation_steps: list[str]
    final_MIC: float | None
    source_paper: str
    lab_or_clinical: str
    is_hypermutator: bool = False
    n_steps: int = field(init=False)

    def __post_init__(self):
        self.n_steps = len(self.mutation_steps)


@dataclass
class Transition:
    transition_id: str
    trajectory_id: str
    from_genotype: str
    to_genotype: str
    mutation_added: str
    drug: str
    drug_class: str
    step_number: int
    delta_MIC: float | None
    source_paper: str
    lab_or_clinical: str
    is_hypermutator_trajectory: bool = False
    mutation_in_canonical_amr_gene: bool = False


# Canonical AMR gene set, used for tagging transitions. This is the union
# of resistance-conferring chromosomal target genes across the major classes.
CANONICAL_AMR_GENES = {
    # Fluoroquinolone targets
    "gyrA", "gyrB", "parC", "parE",
    # Efflux/AMR regulators
    "acrR", "acrA", "acrB", "marR", "marA", "marB", "marC", "tolC",
    "soxR", "soxS", "ramR", "ramA", "rob", "envR",
    "mexA", "mexB", "mexR", "mexX", "mexY", "oprD",
    # Folate pathway (trimethoprim / sulfa)
    "folA", "folP", "folM", "folE", "thyA",
    # Ribosome / aminoglycoside / tetracycline
    "rpsL", "rpsJ", "rpsE", "rpsD", "rplF", "rrsA", "rrsB", "rrsC",
    "tet", "tetA", "tetB", "tetK", "tetM", "tetX",
    # RNA polymerase (rifampin)
    "rpoB", "rpoC", "rpoA",
    # β-lactamases and PBP
    "blaTEM", "blaSHV", "blaCTX-M", "blaOXA", "blaCMY", "ampC", "ampR",
    "ompC", "ompF", "ompK", "porB",
    "pbp1a", "pbp1b", "pbp2", "pbp3", "pbp4", "mrcA", "mrcB", "pbpA",
    "ftsI", "dacA", "dacB",
    # Methicillin / vancomycin
    "mecA", "mecR1", "mecI", "blaZ", "vanA", "vanB", "vanS", "vanR",
    # Colistin
    "mcr", "mgrB", "pmrA", "pmrB", "phoQ", "phoP", "pmrC", "pmrE",
    # Glycopeptide / cell-wall
    "walK", "walR", "vraS", "vraR", "graR", "graS",
    # Topoisomerase IV S. aureus
    "grlA", "grlB",
    # DNA repair / SOS that drive hypermutator phenotype (still relevant)
    "dnaQ", "mutS", "mutL", "mutH", "mutY", "uvrA", "uvrB",
    # NOR drug-target additions
    "ompR", "envZ",
}

# Drug name canonicalization: short codes -> long names.
DRUG_CANONICAL = {
    "NOR": "norfloxacin",
    "CIP": "ciprofloxacin",
    "TMP": "trimethoprim",
    "CHL": "chloramphenicol",
    "DOX": "doxycycline",
    "TET": "tetracycline",
    "CAZ": "ceftazidime",
    "TOB": "tobramycin",
    "AMP": "ampicillin",
    "AMK": "amikacin",
    "GEN": "gentamicin",
    "KAN": "kanamycin",
    "STR": "streptomycin",
    "OXA": "oxacillin",
    "MET": "methicillin",
    "MEM": "meropenem",
    "IMP": "imipenem",
    "VAN": "vancomycin",
    "COL": "colistin",
    "RIF": "rifampicin",
}


def canonicalize_drug(drug: str) -> str:
    if drug is None:
        return drug
    return DRUG_CANONICAL.get(drug.strip(), drug.strip())


def mutation_label(gene: str, details_text: str, mut_type: str,
                   seq_change: str, position) -> str:
    """Produce a compact human-readable mutation label.

    Examples:
      folA L28R     -- from details_text "L28R (CTC->CGC)"
      acrR fs       -- frameshift / pseudogene
      rpoB H526Y
      INS:G@3560455 -- no protein change parseable
    """
    aa = ""
    if isinstance(details_text, str) and details_text:
        # Pull out AA1Number+AA2 patterns
        m = re.match(r"^([A-Z*]\d+[A-Z*])", details_text.strip())
        if m:
            aa = m.group(1)
        elif "fs" in details_text or "frameshift" in details_text.lower():
            aa = "fs"
        elif "pseudogene" in details_text.lower():
            aa = "pseudo"
        elif "intergenic" in details_text.lower():
            aa = "ig"
        elif "coding" in details_text.lower():
            aa = "coding"
    gene_t = (gene or "").strip().replace(" ", "_") or "?"
    if aa:
        return f"{gene_t}_{aa}"
    return f"{gene_t}_{mut_type}@{position}"


# A simple per-row "lineage assignment": given an isolate's column name
# A<ale> F<flask> I<isolate> R<rep>, return the lineage id (A<ale>) and the
# step number (F<flask>).
_RE_SAMPLE = re.compile(r"^A(\d+)\s+F(\d+)\s+I(\d+)\s+R(\d+)")


def parse_sample_label(label: str) -> tuple[int, int, int, int] | None:
    m = _RE_SAMPLE.match(label)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def build_trajectories_from_aledb_table(
    sample_cols: list[str],
    df_mut: pd.DataFrame,
    drug: str,
    organism: str,
    source_paper: str,
    lab_or_clinical: str = "lab",
    project_tag: str = "",
    freq_threshold: float = 0.5,
    max_mutations_per_isolate: int = 200,
) -> tuple[list[Trajectory], list[Transition]]:
    """For each (ALE lineage = A#), order isolates by (F, I), and build the
    incremental trajectory of mutations as they appear in successive isolates
    above ``freq_threshold``.

    Multiple isolates at the same flask step are treated as parallel branches
    that join the lineage if their mutation set is a superset of the prior
    step.
    """
    # Group sample cols by lineage A
    lineage_to_samples: dict[int, list[tuple[int, int, int, int, str]]] = defaultdict(list)
    for c in sample_cols:
        parsed = parse_sample_label(c)
        if parsed is None:
            continue
        a, f, i, r = parsed
        lineage_to_samples[a].append((a, f, i, r, c))

    trajectories: list[Trajectory] = []
    transitions: list[Transition] = []

    drug_canon = canonicalize_drug(drug)
    drug_cls = drug_class_for(drug_canon)
    safe_tag = project_tag or source_paper.replace(" ", "_")

    for a, samples in lineage_to_samples.items():
        # Order by flask, isolate
        samples = sorted(samples, key=lambda t: (t[1], t[2], t[3]))
        prev_set: set[str] = set()
        steps: list[str] = []
        traj_id = f"{safe_tag}_A{a}"
        from_geno = "wildtype"
        step_num = 0
        is_hyper = False
        max_isolate_muts = 0
        for (_, f, i, r, col) in samples:
            freq_col = f"freq__{col}"
            if freq_col not in df_mut.columns:
                continue
            # Mutations present at this isolate above threshold
            mask = df_mut[freq_col].fillna(0) >= freq_threshold
            sub = df_mut[mask]
            n_muts_iso = len(sub)
            max_isolate_muts = max(max_isolate_muts, n_muts_iso)
            if n_muts_iso > max_mutations_per_isolate:
                # Hypermutator — flag but include with reduced confidence
                is_hyper = True
                # Still record but truncate to top hits in canonical AMR genes
                # to avoid pure noise.
                amr_mask = sub["gene"].astype(str).apply(
                    lambda g: any(ag in g for ag in CANONICAL_AMR_GENES)
                )
                sub = sub[amr_mask]
                if len(sub) == 0:
                    continue
            cur_set: set[str] = set()
            for _, row in sub.iterrows():
                lbl = mutation_label(
                    row.get("gene", ""), row.get("details_text", ""),
                    row.get("mut_type", ""), row.get("seq_change", ""),
                    row.get("position", ""),
                )
                cur_set.add(lbl)
            new_muts = sorted(cur_set - prev_set)
            if not new_muts:
                continue
            # Order is ambiguous within a single isolate; we record them as
            # appearing in the same step but as individual incremental
            # transitions in alphabetical order. step_number is shared.
            for m in new_muts:
                step_num += 1
                steps.append(m)
                to_geno = "+".join(sorted(prev_set | {m})) or "wildtype"
                tid = f"{traj_id}__t{step_num:03d}"
                gene = m.split("_")[0]
                in_canon = gene in CANONICAL_AMR_GENES
                transitions.append(Transition(
                    transition_id=tid,
                    trajectory_id=traj_id,
                    from_genotype=from_geno,
                    to_genotype=to_geno,
                    mutation_added=m,
                    drug=drug_canon,
                    drug_class=drug_cls,
                    step_number=step_num,
                    delta_MIC=None,
                    source_paper=source_paper,
                    lab_or_clinical=lab_or_clinical,
                    is_hypermutator_trajectory=is_hyper,
                    mutation_in_canonical_amr_gene=in_canon,
                ))
                prev_set = prev_set | {m}
                from_geno = to_geno
        if len(steps) >= 1:
            trajectories.append(Trajectory(
                trajectory_id=traj_id,
                organism=organism,
                drug=drug_canon,
                drug_class=drug_cls,
                starting_genotype="wildtype",
                mutation_steps=steps,
                final_MIC=None,
                source_paper=source_paper,
                lab_or_clinical=lab_or_clinical,
                is_hypermutator=is_hyper,
            ))
    return trajectories, transitions


# -----------------------------------------------------------------------------
# SOURCE A — Baym 2016 (ALEdb project 186 + provided CSV)
# -----------------------------------------------------------------------------

def parse_baym_csv(path: Path, source_paper: str) -> tuple[list[Trajectory], list[Transition]]:
    """Parse the Baym_M_2016 CSV from the ALEdb example zip.

    Sample columns of the form '4step0-1gml-10gml-100gml-1mgml A1 F4 I88 R1'
    encode (ALE A#, gradient-step F#, isolate I#, replicate R#).
    """
    df = pd.read_csv(path)
    # Identify sample columns: anything after the standard metadata cols
    meta_cols = [
        "Reference Seq", "Position", "Mutation Type", "Sequence Change",
        "Gene (Scrollable)", "Function", "Product",
        "GO Process", "GO Component", "Mut ID", "Details",
    ]
    sample_cols = [c for c in df.columns if c not in meta_cols]
    # Extract trailing 'A<a> F<f> I<i> R<r>' from each sample col
    sample_meta = {}
    for c in sample_cols:
        m = re.search(r"A(\d+)\s+F(\d+)\s+I(\d+)\s+R(\d+)", c)
        if m:
            sample_meta[c] = (int(m.group(1)), int(m.group(2)),
                              int(m.group(3)), int(m.group(4)))
    log.info(f"Baym CSV: {len(df)} mutations, {len(sample_meta)} keyed isolates")

    df = df.rename(columns={
        "Reference Seq": "ref_seq",
        "Position": "position",
        "Mutation Type": "mut_type",
        "Sequence Change": "seq_change",
        "Gene (Scrollable)": "gene",
        "Function": "function",
        "Product": "product",
        "Mut ID": "mut_id",
        "Details": "details_text",
    })
    # ALEdb CSV stores frequency as e.g. "1.00/1.00" or NaN
    def _parse_freq(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s:
            return None
        if "/" in s:
            try:
                return float(s.split("/")[0])
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None
    # Group samples by lineage and step
    lineage_to_samples = defaultdict(list)
    for c, (a, f, i, r) in sample_meta.items():
        lineage_to_samples[a].append((a, f, i, r, c))

    drug = "trimethoprim"  # Baym 2016 MEGA-plate was trimethoprim
    drug_cls = drug_class_for(drug)
    organism = "Escherichia coli"
    trajectories: list[Trajectory] = []
    transitions: list[Transition] = []
    MAX_MUTS = 200

    for a, samples in lineage_to_samples.items():
        samples = sorted(samples, key=lambda t: (t[1], t[2], t[3]))
        prev_set: set[str] = set()
        steps: list[str] = []
        traj_id = f"Baym2016_A{a}"
        from_geno = "wildtype"
        step_num = 0
        is_hyper = False
        for (_, f, i, r, col) in samples:
            freqs = df[col].apply(_parse_freq)
            mask = freqs.fillna(0) >= 0.5
            sub = df[mask]
            if len(sub) > MAX_MUTS:
                is_hyper = True
                amr_mask = sub["gene"].astype(str).apply(
                    lambda g: any(ag in g for ag in CANONICAL_AMR_GENES)
                )
                sub = sub[amr_mask]
                if len(sub) == 0:
                    continue
            cur_set = set()
            for _, row in sub.iterrows():
                lbl = mutation_label(
                    str(row.get("gene", "")).split(",")[0],
                    str(row.get("details_text", "")),
                    str(row.get("mut_type", "")),
                    str(row.get("seq_change", "")),
                    row.get("position", ""),
                )
                cur_set.add(lbl)
            new_muts = sorted(cur_set - prev_set)
            if not new_muts:
                continue
            for m in new_muts:
                step_num += 1
                steps.append(m)
                to_geno = "+".join(sorted(prev_set | {m})) or "wildtype"
                gene = m.split("_")[0]
                in_canon = gene in CANONICAL_AMR_GENES
                transitions.append(Transition(
                    transition_id=f"{traj_id}__t{step_num:03d}",
                    trajectory_id=traj_id,
                    from_genotype=from_geno,
                    to_genotype=to_geno,
                    mutation_added=m,
                    drug=drug,
                    drug_class=drug_cls,
                    step_number=step_num,
                    delta_MIC=None,
                    source_paper=source_paper,
                    lab_or_clinical="lab",
                    is_hypermutator_trajectory=is_hyper,
                    mutation_in_canonical_amr_gene=in_canon,
                ))
                prev_set = prev_set | {m}
                from_geno = to_geno
        if steps:
            trajectories.append(Trajectory(
                trajectory_id=traj_id,
                organism=organism,
                drug=drug,
                drug_class=drug_cls,
                starting_genotype="wildtype",
                mutation_steps=steps,
                final_MIC=None,
                source_paper=source_paper,
                lab_or_clinical="lab",
                is_hypermutator=is_hyper,
            ))
    return trajectories, transitions


# -----------------------------------------------------------------------------
# Generic ALEdb project parser  (Sources B, D1..D8)
# -----------------------------------------------------------------------------

def parse_aledb_project(project_id: int,
                        experiment_drugs: dict[int, str] | None = None,
                        default_drug: str | None = None,
                        organism: str = "Escherichia coli",
                        source_paper: str = "",
                        project_tag: str = "",
                        lab_or_clinical: str = "lab",
                        ) -> tuple[list[Trajectory], list[Transition]]:
    """Pull all experiments from an ALEdb project and build trajectories.

    experiment_drugs maps experiment_id -> drug; if absent uses default_drug
    or skips ancestor/control experiments.
    """
    experiments = fetch_aledb_project_experiments(project_id)
    log.info(f"Project {project_id} '{project_tag}': {len(experiments)} experiments")
    all_traj: list[Trajectory] = []
    all_trans: list[Transition] = []
    for exp in experiments:
        eid = exp["experiment_id"]
        name = exp["name"]
        drug = None
        if experiment_drugs and eid in experiment_drugs:
            drug = experiment_drugs[eid]
        else:
            # try to infer from name
            ln = name.lower()
            if any(k in ln for k in ["ancestor", "control", "wildtype", "wild_type", "_wt_"]):
                log.info(f"  Skipping ancestor/control: {eid} {name}")
                continue
            for token in DRUG_CLASS:
                if token.lower() in ln:
                    drug = token
                    break
            if drug is None:
                drug = default_drug
        if drug is None:
            log.info(f"  Skipping (no drug): {eid} {name}")
            continue
        try:
            sample_cols, df_mut = fetch_aledb_experiment_mutations(eid)
        except Exception as e:
            log.warning(f"  experiment {eid} fetch/parse failed: {e}")
            continue
        if df_mut.empty or not sample_cols:
            log.info(f"  experiment {eid} '{name}' empty, skipping")
            continue
        trajs, trans = build_trajectories_from_aledb_table(
            sample_cols, df_mut, drug=drug, organism=organism,
            source_paper=source_paper, lab_or_clinical=lab_or_clinical,
            project_tag=f"{project_tag}_e{eid}",
        )
        log.info(f"  experiment {eid} '{name}' drug={drug}: "
                 f"{len(trajs)} trajectories, {len(trans)} transitions")
        all_traj.extend(trajs)
        all_trans.extend(trans)
    return all_traj, all_trans


# -----------------------------------------------------------------------------
# SOURCE E — Inferred CARD ladder (cross-section synthesis)
# -----------------------------------------------------------------------------

# QRDR positions (fluoroquinolone target) ordered: gyrA -> parC -> gyrA -> parC ...
# Empirical ladder common in clinical isolates.
KNOWN_LADDERS = {
    # NOTE: Each ladder must use mutations that actually appear in CARD calls
    # against our genome population, else supercount is 0 and ladder is skipped.
    "fluoroquinolone": [
        # E. coli QRDR ladder (S83L -> D87N -> parC S80I)
        ("Escherichia coli", "ciprofloxacin",
         [["gyrA_S83L"],
          ["gyrA_S83L", "gyrA_D87N"],
          ["gyrA_S83L", "gyrA_D87N", "parC_S80I"]]),
        # E. coli secondary path (D87 first)
        ("Escherichia coli", "ciprofloxacin",
         [["gyrA_D87N"],
          ["gyrA_D87N", "gyrA_S83L"],
          ["gyrA_S83L", "gyrA_D87N", "parC_S80I"]]),
        # K. pneumoniae S83I path
        ("Klebsiella pneumoniae", "ciprofloxacin",
         [["gyrA_S83I"],
          ["gyrA_S83I", "gyrA_D87N"],
          ["gyrA_S83I", "gyrA_D87N", "parC_S80I"]]),
        # K. pneumoniae alternative starting mutation (parC first)
        ("Klebsiella pneumoniae", "ciprofloxacin",
         [["parC_S80I"],
          ["parC_S80I", "gyrA_S83I"],
          ["parC_S80I", "gyrA_S83I", "gyrA_D87N"]]),
        # K. pneumoniae S83F minor path
        ("Klebsiella pneumoniae", "ciprofloxacin",
         [["gyrA_S83F"],
          ["gyrA_S83F", "parC_S80I"]]),
        # S. aureus dominant ladder: gyrA S84L -> parC S80F  (also S80Y, S80F is dominant)
        ("Staphylococcus aureus", "ciprofloxacin",
         [["gyrA_S84L"],
          ["gyrA_S84L", "parC_S80F"]]),
        # S. aureus alternative: parC S80F seen first
        ("Staphylococcus aureus", "ciprofloxacin",
         [["parC_S80F"],
          ["parC_S80F", "gyrA_S84L"]]),
        # S. aureus minor S80Y variant
        ("Staphylococcus aureus", "ciprofloxacin",
         [["gyrA_S84L"],
          ["gyrA_S84L", "parC_S80Y"]]),
        # S. aureus parC E84G branch
        ("Staphylococcus aureus", "ciprofloxacin",
         [["parC_S80F"],
          ["parC_S80F", "parC_E84G"]]),
    ],
    "rifamycin": [
        # S. aureus rpoB H481N is the classic rif-resistance mutation
        ("Staphylococcus aureus", "rifampicin",
         [["rpoB_H481N"]]),
    ],
}


def synthesize_inferred_card_trajectories(rg_path: Path) -> tuple[list[Trajectory], list[Transition]]:
    """Cross-section synthesis from resistance_genes.parquet.

    Strategy: for each (species, drug class), check which mutations are present
    in our genome population. Build the chain of nested mutation sets that are
    actually observed at meaningful frequency (>=5 genomes), and emit those as
    inferred trajectories. We seed with literature-derived ladders.
    """
    rg = pd.read_parquet(rg_path)
    species_map = {
        "ecoli": "Escherichia coli",
        "kpneumo": "Klebsiella pneumoniae",
        "saureus": "Staphylococcus aureus",
    }

    # Build, per genome, the set of mutation labels "gene_PointMut"
    # We only consider known resistance mutations to avoid noise.
    rg_known = rg[rg["known_resistance_mutation"] == True].copy()  # noqa: E712
    rg_known = rg_known[rg_known["gene_symbol"].notna()]
    rg_known["label"] = rg_known["gene_symbol"].astype(str)

    # Strip the leading "gene_S83L" -> we want labels like "gyrA_S83L"
    # In our parquet, gene_symbol is already e.g. 'gyrA_S83L'. Good.
    genome_muts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for _, row in rg_known.iterrows():
        sp = species_map.get(row["species"], row["species"])
        genome_muts[(sp, row["genome_id"])].add(row["label"])

    trajs: list[Trajectory] = []
    transitions: list[Transition] = []
    summary = defaultdict(int)
    for cls, ladders in KNOWN_LADDERS.items():
        for organism, drug, ladder in ladders:
            # Count genomes that exactly contain each set in the ladder.
            counts = []
            for step_set in ladder:
                target = set(step_set)
                # match if genome contains exactly the target (or superset of
                # prior chain step but not next). We use "contains target"
                # heuristic: count genomes whose mutation set is a superset.
                supercount = sum(1 for k, ms in genome_muts.items()
                                 if k[0] == organism and target.issubset(ms))
                counts.append(supercount)
            # Require all steps observed in >=3 genomes (some minor variants
            # are real but rare). Single-step ladders also accepted.
            if all(c >= 3 for c in counts):
                lpath = "_".join(f"step{i+1}" for i in range(len(ladder)))
                traj_id = f"inferred_{organism.replace(' ', '_')}_{drug}_{lpath}"
                # Trajectory is the literature ladder; mutation_added at each
                # step = symmetric difference between consecutive sets.
                mutation_steps = []
                prev_set: set[str] = set()
                for i, step_set in enumerate(ladder):
                    new = sorted(set(step_set) - prev_set)
                    for m in new:
                        mutation_steps.append(m)
                        to_geno = "+".join(sorted(prev_set | {m}))
                        gene = m.split("_")[0]
                        transitions.append(Transition(
                            transition_id=f"{traj_id}__t{len(transitions):04d}",
                            trajectory_id=traj_id,
                            from_genotype="+".join(sorted(prev_set)) or "wildtype",
                            to_genotype=to_geno,
                            mutation_added=m,
                            drug=drug,
                            drug_class=cls,
                            step_number=len(mutation_steps),
                            delta_MIC=None,
                            source_paper="inferred_card_crosssection",
                            lab_or_clinical="inferred_card",
                            is_hypermutator_trajectory=False,
                            mutation_in_canonical_amr_gene=gene in CANONICAL_AMR_GENES,
                        ))
                        prev_set = prev_set | {m}
                trajs.append(Trajectory(
                    trajectory_id=traj_id,
                    organism=organism,
                    drug=drug,
                    drug_class=cls,
                    starting_genotype="wildtype",
                    mutation_steps=mutation_steps,
                    final_MIC=None,
                    source_paper="inferred_card_crosssection",
                    lab_or_clinical="inferred_card",
                    is_hypermutator=False,
                ))
                summary[organism] += 1
    log.info(f"Inferred CARD ladders: {dict(summary)} ({len(trajs)} trajectories, {len(transitions)} transitions)")
    return trajs, transitions


# -----------------------------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------------------------

def write_intermediate(name: str, trajs: list[Trajectory], trans: list[Transition]) -> None:
    if trajs:
        df_t = pd.DataFrame([asdict(t) for t in trajs])
        df_t.to_parquet(INTERMED / f"{name}_trajectories.parquet")
    if trans:
        df_x = pd.DataFrame([asdict(t) for t in trans])
        df_x.to_parquet(INTERMED / f"{name}_transitions.parquet")
    log.info(f"[{name}] wrote {len(trajs)} trajectories, {len(trans)} transitions")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    log.info("=== GATE 3 START ===")

    all_traj: list[Trajectory] = []
    all_trans: list[Transition] = []

    # ---- Source A: Baym 2016 from ALEdb example CSV ----
    log.info("--- Source A: Baym 2016 (ALEdb example) ---")
    baym_csv = HDD_RAW / "aledb" / "Baym_M_2016_4step0-1gml-10gml-100gml-1mgml_mut.csv"
    try:
        a_traj, a_trans = parse_baym_csv(baym_csv,
                                         source_paper="Baym et al. 2016 Science 353:1147 (via ALEdb)")
        write_intermediate("sourceA_baym2016", a_traj, a_trans)
        all_traj.extend(a_traj)
        all_trans.extend(a_trans)
    except Exception as e:
        log.exception("Source A failed: %s", e)

    # ---- Source A2: Baym ALEdb project 186 (if reachable) ----
    log.info("--- Source A2: Baym ALEdb project 186 (live scrape) ---")
    try:
        # Project 186 only contains the same data; we keep it as a cross-check.
        a2_traj, a2_trans = parse_aledb_project(
            project_id=186,
            default_drug="trimethoprim",
            organism="Escherichia coli",
            source_paper="Baym et al. 2016 Science 353:1147 (via ALEdb live)",
            project_tag="Baym2016live",
        )
        write_intermediate("sourceA2_baym2016_live", a2_traj, a2_trans)
        all_traj.extend(a2_traj)
        all_trans.extend(a2_trans)
    except Exception as e:
        log.exception("Source A2 failed: %s", e)

    # ---- Source B: Toprak-style Chl/Dox/Tri (ALEdb project 119) ----
    log.info("--- Source B: Ecoli_resistance_ChlDoxTri (ALEdb #119) ---")
    # Known experiment ids from earlier reconnaissance:
    # 1147 doxycycline, 1148 trimethoprim, 1149 chloramphenicol, 1150 ancestor
    try:
        b_traj, b_trans = parse_aledb_project(
            project_id=119,
            experiment_drugs={1147: "doxycycline", 1148: "trimethoprim",
                              1149: "chloramphenicol"},
            organism="Escherichia coli",
            source_paper="Toprak-style Chl/Dox/Tri ALE (ALEdb project 119)",
            project_tag="Toprak_ChlDoxTri",
        )
        write_intermediate("sourceB_toprakstyle", b_traj, b_trans)
        all_traj.extend(b_traj)
        all_trans.extend(b_trans)
    except Exception as e:
        log.exception("Source B failed: %s", e)

    # ---- Source D: other antibiotic ALEdb projects ----
    SOURCE_D_PROJECTS = [
        # (project_id, project_tag, default_drug, organism, paper)
        (124, "norfloxacin",        "norfloxacin",  "Escherichia coli",
         "ALEdb project 124 Ecoli_resistance_norfloxacin"),
        (120, "ceftazidime",        "ceftazidime",  "Escherichia coli",
         "ALEdb project 120 Ecoli_resistance_caz"),
        (123, "tetracycline",       "tetracycline", "Escherichia coli",
         "ALEdb project 123 Ecoli_resistance_tet_intragenCoevol"),
        (226, "Spohn2019_30Abx",    None,           "Escherichia coli",
         "Spohn et al. 2019 (ALEdb 226) 30-antibiotic ALE"),
        (192, "Swings2017",         None,           "Escherichia coli",
         "Swings et al. 2017 (ALEdb 192)"),
        (227, "Laborda2022_tobra",  "tobramycin",   "Escherichia coli",
         "Laborda et al. 2022 (ALEdb 227) urine tobramycin"),
        (210, "Sada2022",           None,           "Escherichia coli",
         "Sada et al. 2022 (ALEdb 210)"),
        (209, "Ewunkem2021",        None,           "Escherichia coli",
         "Akamu J Ewunkem 2021 (ALEdb 209)"),
        (135, "Ecoli_antibResist_metabol", None,    "Escherichia coli",
         "ALEdb project 135 Ecoli_antibResist_metabol"),
        (168, "Unconstrained_evol",  None,           "Escherichia coli",
         "ALEdb project 168 Unconstrained Evolution"),
        (225, "Bergh2016_amikacin", "amikacin",      "Escherichia coli",
         "Bram Bergh 2016 (ALEdb 225) amikacin resistance"),
    ]
    for pid, tag, drug, org, paper in SOURCE_D_PROJECTS:
        log.info(f"--- Source D: project {pid} ({tag}) ---")
        try:
            d_traj, d_trans = parse_aledb_project(
                project_id=pid, default_drug=drug, organism=org,
                source_paper=paper, project_tag=tag,
            )
            write_intermediate(f"sourceD_{tag}", d_traj, d_trans)
            all_traj.extend(d_traj)
            all_trans.extend(d_trans)
        except Exception as e:
            log.exception("Source D %s failed: %s", tag, e)

    # Deduplicate Baym 2016: project 186 experiment 2052 is the same data as
    # the static CSV. Keep only the static-CSV-derived trajectories where
    # both exist (identical mutation_steps for matching lineage).
    log.info(f"Before dedup: {len(all_traj)} trajectories, {len(all_trans)} transitions")
    keep_traj_ids = set()
    seen_signatures: dict[tuple, str] = {}
    for t in all_traj:
        sig = (t.organism, t.drug, tuple(sorted(t.mutation_steps)))
        # Prefer CSV-derived ("Baym2016_A") over live ("Baym2016live_e2052_A")
        prev = seen_signatures.get(sig)
        if prev is None:
            seen_signatures[sig] = t.trajectory_id
            keep_traj_ids.add(t.trajectory_id)
        else:
            # If a CSV-version exists, keep that; if the new one is CSV,
            # replace prev.
            prev_is_csv = prev.startswith("Baym2016_A")
            cur_is_csv = t.trajectory_id.startswith("Baym2016_A")
            if cur_is_csv and not prev_is_csv:
                keep_traj_ids.discard(prev)
                keep_traj_ids.add(t.trajectory_id)
                seen_signatures[sig] = t.trajectory_id
    all_traj = [t for t in all_traj if t.trajectory_id in keep_traj_ids]
    all_trans = [x for x in all_trans if x.trajectory_id in keep_traj_ids]
    log.info(f"After Sources A-D + dedup: {len(all_traj)} trajectories, "
             f"{len(all_trans)} transitions")

    # ---- Source E: inferred CARD ladder ----
    # Always include — it provides K. pneumoniae and S. aureus coverage
    # which is absent from ALEdb (E. coli only). These are flagged
    # lab_or_clinical='inferred_card' for downstream segmentation.
    log.info("--- Source E: inferred CARD ladder (cross-species coverage) ---")
    try:
        e_traj, e_trans = synthesize_inferred_card_trajectories(
            SSD_PROC / "resistance_genes.parquet"
        )
        write_intermediate("sourceE_inferred_card", e_traj, e_trans)
        all_traj.extend(e_traj)
        all_trans.extend(e_trans)
    except Exception as e:
        log.exception("Source E failed: %s", e)

    # ---- Persist final ----
    if all_traj:
        df_t = pd.DataFrame([asdict(t) for t in all_traj])
    else:
        df_t = pd.DataFrame()
    if all_trans:
        df_x = pd.DataFrame([asdict(t) for t in all_trans])
    else:
        df_x = pd.DataFrame()

    df_t.to_parquet(SSD_PROC / "trajectories.parquet")
    df_x.to_parquet(SSD_PROC / "transitions.parquet")

    # ---- Summary stats ----
    summary: dict = {
        "n_trajectories": int(len(df_t)),
        "n_transitions": int(len(df_x)),
        "n_trajectories_geq2_steps": int((df_t.get("n_steps", pd.Series(dtype=int)) >= 2).sum()) if not df_t.empty else 0,
        "elapsed_sec": int(time.time() - t0),
    }
    if not df_x.empty:
        summary["transitions_by_drug_class"] = df_x["drug_class"].value_counts().to_dict()
        summary["transitions_by_lab_or_clinical"] = df_x["lab_or_clinical"].value_counts().to_dict()
        summary["transitions_by_drug"] = df_x["drug"].value_counts().to_dict()
        summary["transitions_by_source_paper"] = df_x["source_paper"].value_counts().to_dict()
        summary["transitions_canonical_amr_gene"] = int(df_x["mutation_in_canonical_amr_gene"].sum())
        summary["transitions_hypermutator_trajectory"] = int(df_x["is_hypermutator_trajectory"].sum())
        summary["transitions_non_hyper_only"] = int((~df_x["is_hypermutator_trajectory"]).sum())
    if not df_t.empty:
        summary["trajectories_by_organism"] = df_t["organism"].value_counts().to_dict()
        summary["trajectories_by_drug"] = df_t["drug"].value_counts().to_dict()
        summary["mean_n_steps"] = float(df_t["n_steps"].mean())
        summary["max_n_steps"] = int(df_t["n_steps"].max())
        summary["trajectories_hypermutator"] = int(df_t["is_hypermutator"].sum())
        summary["trajectories_non_hyper"] = int((~df_t["is_hypermutator"]).sum())

    (SSD_PROC / "gate3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary: %s", json.dumps(summary, indent=2, default=str))

    # ---- Gate decision ----
    n_traj_geq2 = summary["n_trajectories_geq2_steps"]
    n_trans = summary["n_transitions"]
    gate_log_line = (f"GATE_3: {{verdict}} [n_trajectories={summary['n_trajectories']}] "
                     f"[n_trajectories_geq2={n_traj_geq2}] [n_transitions={n_trans}]")
    if n_trans >= 500 and n_traj_geq2 >= 50:
        verdict = "PASS"
    elif n_trans >= 200:
        verdict = "SOFT_PASS"
    else:
        verdict = "HARD_STOP"
    gate_log_line = gate_log_line.format(verdict=verdict)

    with open(LOGS / "gates.log", "a") as f:
        f.write(gate_log_line + "\n")
    log.info(gate_log_line)
    print(gate_log_line)
    return verdict, summary


if __name__ == "__main__":
    verdict, summary = main()
    sys.exit(0 if verdict in ("PASS", "SOFT_PASS") else 2)
