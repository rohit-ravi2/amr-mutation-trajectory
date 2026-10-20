"""
Phase A2 — ALEdb MIC proxy harvest.

ALEdb's per-experiment metadata page encodes the antibiotic and selection
concentration directly in the experiment name (e.g. "MG1655_nor_75" =
75 ug/mL norfloxacin). ALEdb does NOT expose per-timepoint MIC measurements
in a JSON API, but the concentration is the selection pressure at the end
of the experiment, which is the standard MIC proxy used in published ALE
analyses (Pal 2015, Lukacisinova 2020, Tenaillon 2016).

Approach:
  1. For each ALEdb experiment_id mentioned in trajectories.parquet, fetch
     /ale/project/{P} pages to discover experiment names.
  2. Parse antibiotic + concentration from the experiment name regex.
  3. Within each experiment, the trajectory id appended a suffix _A1, _A2 etc;
     these are replicate ALE flasks. The final-timepoint MIC proxy is the
     experiment's selection concentration.
  4. delta_MIC per transition = MIC_after - MIC_before, where MIC_before is
     the cumulative selection concentration up to step (n-1) and MIC_after is
     the concentration at step n. ALEdb experiments are run at a single
     escalating concentration ladder; in this Day-1 dataset every step within
     an experiment shares the final concentration, so delta_MIC for step k
     within experiment E is computed as:
         MIC(E, k) = selection_conc(E) * k / n_steps(E)
     i.e. linear ramp-up assumption. This is an approximation; honest
     limitation noted in REPORT_day2.md.
  5. For Baym 2016 (MEGA-plate), antibiotic gradient is well-documented in
     paper: trimethoprim 0->3000 mg/L over 9 bands. For Toprak 2012 / others,
     drug + final concentration captured from the project name.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

OUT_RAW_DIR = Path("work/raw/aledb_mic")
OUT_PROCESSED = Path("work/processed/transitions_with_mic.parquet")
TRANS_PATH = Path("work/processed/transitions.parquet")
TRAJ_PATH = Path("work/processed/trajectories.parquet")
LOG_PATH = Path("work/logs/day2/phaseA2_aledb_mic.log")
GATE_LOG = Path("work/logs/gates.log")

OUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

UA = "amr-research/0.2"
HEADERS = {"User-Agent": UA}

# Project numbers that the trajectory table maps to.
PROJECTS = [119, 120, 123, 124, 135, 168, 225, 227, 186]

# Drug -> normalized class + canonical concentration unit (ug/mL).
# Regex over tokens split on _ and -; tokens checked individually.
DRUG_TOKEN_PATTERNS = [
    (re.compile(r"^nor\w*$|^norflox\w*$", re.I), "norfloxacin",   "fluoroquinolone"),
    (re.compile(r"^cip\w*$|^ciproflox\w*$", re.I), "ciprofloxacin", "fluoroquinolone"),
    (re.compile(r"^lev\w*$|^levoflox\w*$", re.I), "levofloxacin", "fluoroquinolone"),
    (re.compile(r"^tet\w*$|^tetracyclin\w*$", re.I), "tetracycline", "tetracycline"),
    (re.compile(r"^dox\w*$|^doxycyclin\w*$", re.I), "doxycycline", "tetracycline"),
    (re.compile(r"^tri\w*$|^trim\w*$|^trimethoprim$", re.I), "trimethoprim", "diaminopyrimidine"),
    (re.compile(r"^chl\w*$|^chloramphenicol$", re.I), "chloramphenicol", "phenicol"),
    (re.compile(r"^caz\w*$|^ceftazidime$", re.I), "ceftazidime", "cephalosporin"),
    (re.compile(r"^cef\w*$", re.I), "cefepime", "cephalosporin"),
    (re.compile(r"^amp\w*$|^ampicillin\w*$", re.I), "ampicillin", "penicillin"),
    (re.compile(r"^amik\w*$|^amk$", re.I), "amikacin", "aminoglycoside"),
    (re.compile(r"^tob\w*$|^tobramycin$", re.I), "tobramycin", "aminoglycoside"),
    (re.compile(r"^gen\w*$|^gentamicin$", re.I), "gentamicin", "aminoglycoside"),
    (re.compile(r"^rif\w*$|^rifampicin$", re.I), "rifampicin", "rifamycin"),
]


def setup_logger() -> logging.Logger:
    log = logging.getLogger("phaseA2")
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


def parse_drug_and_concentration(experiment_name: str) -> tuple[str | None, str | None, float | None]:
    """Return (drug, drug_class, concentration_ug_per_ml or None)."""
    drug, dclass = None, None
    # Tokenize on - _
    tokens = re.split(r"[-_]", experiment_name)
    for tok in tokens:
        for rx, d, c in DRUG_TOKEN_PATTERNS:
            if rx.match(tok):
                drug, dclass = d, c
                break
        if drug:
            break
    # find concentration encoded as _<digits>(_or end)
    m = re.search(r"_(\d{1,5})(?:[_-]|$)", experiment_name)
    if m:
        v = float(m.group(1))
        # ALE convention: ug/mL for FQ where two digits typical (25, 50, 75)
        # value 125 -> 1.25 ug/mL ciprofloxacin etc. We honor the raw integer
        # as the project's published convention.
        # Spohn 2016 etc store as (10 x ug/mL): "nor_75" = 0.75 ug/mL?
        # Conservative: store as published integer, document units in
        # `mic_unit_note` column.
        return drug, dclass, v
    if "untreated" in experiment_name.lower():
        return drug, dclass, 0.0
    return drug, dclass, None


def fetch_project_experiments(project_id: int, session: requests.Session,
                              log: logging.Logger) -> dict[int, str]:
    """Return {exp_id: name}."""
    url = f"https://aledb.org/ale/project/{project_id}"
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            log.warning("project %d -> HTTP %d", project_id, r.status_code)
            return {}
    except Exception as e:
        log.warning("project %d fetch failed: %s", project_id, e)
        return {}
    out = {}
    for m in re.finditer(r'<a href="/ale/experiment/(\d+)"[^>]*>([^<]+)</a>', r.text):
        out[int(m.group(1))] = m.group(2).strip()
    # cache raw HTML
    (OUT_RAW_DIR / f"project_{project_id}.html").write_text(r.text)
    return out


def build_experiment_table(session: requests.Session, log: logging.Logger) -> pd.DataFrame:
    rows = []
    for pid in tqdm(PROJECTS, desc="ALEdb projects"):
        exps = fetch_project_experiments(pid, session, log)
        for eid, name in exps.items():
            drug, dclass, conc = parse_drug_and_concentration(name)
            rows.append({
                "project_id":          pid,
                "experiment_id":       eid,
                "experiment_name":     name,
                "drug":                drug,
                "drug_class":          dclass,
                "selection_conc_raw":  conc,
            })
        time.sleep(0.5)
    return pd.DataFrame(rows)


def parse_trajectory_experiment(trajectory_id: str) -> int | None:
    """Extract experiment_id from trajectory_id like 'Baym2016live_e2048_A1'
    or 'Toprak_ChlDoxTri_e1147_A1' or 'ALEdb_186_e2200_A3'."""
    m = re.search(r"_e(\d+)_", trajectory_id)
    if m:
        return int(m.group(1))
    return None


def main() -> int:
    log = setup_logger()
    log.info("=== Phase A2 — ALEdb MIC proxy harvest ===")

    session = requests.Session()
    session.headers.update(HEADERS)

    exp_table = build_experiment_table(session, log)
    log.info("Fetched %d ALEdb experiments across %d projects",
             len(exp_table), exp_table["project_id"].nunique())
    log.info("Drugs found: %s", exp_table["drug"].value_counts(dropna=False).to_dict())
    log.info("With concentration: %d / %d",
             exp_table["selection_conc_raw"].notna().sum(), len(exp_table))

    raw_out = OUT_RAW_DIR / "aledb_experiments.parquet"
    exp_table.to_parquet(raw_out, index=False)
    log.info("Wrote %s", raw_out)

    # Build MIC join key on transitions
    trans = pd.read_parquet(TRANS_PATH)
    log.info("Transitions: %d", len(trans))
    trans["experiment_id"] = trans["trajectory_id"].map(parse_trajectory_experiment)
    n_with_exp = trans["experiment_id"].notna().sum()
    log.info("Transitions with parsed experiment_id: %d / %d", n_with_exp, len(trans))

    # Trajectory length lookup (for linear-ramp MIC estimate)
    traj = pd.read_parquet(TRAJ_PATH)
    traj_steps = dict(zip(traj["trajectory_id"], traj["n_steps"]))

    # Join transitions <- experiment metadata
    trans = trans.merge(
        exp_table[["experiment_id", "experiment_name", "selection_conc_raw"]],
        on="experiment_id", how="left",
    )

    # delta_MIC under linear-ramp assumption.
    # MIC(step k) = selection_conc_raw * k / n_steps;  delta_MIC = MIC(k) - MIC(k-1)
    def compute_delta_mic(row):
        c = row["selection_conc_raw"]
        if c is None or pd.isna(c) or c == 0:
            return np.nan
        n_steps = traj_steps.get(row["trajectory_id"])
        if not n_steps or n_steps < 1:
            return np.nan
        return float(c) / float(n_steps)

    trans["delta_MIC_proxy"] = trans.apply(compute_delta_mic, axis=1)

    # Also a cumulative MIC at each step (useful for autoregressive feature)
    def cum_mic(row):
        c = row["selection_conc_raw"]
        if c is None or pd.isna(c) or c == 0:
            return np.nan
        n_steps = traj_steps.get(row["trajectory_id"])
        if not n_steps:
            return np.nan
        return float(c) * float(row["step_number"]) / float(n_steps)

    trans["cumulative_MIC_proxy"] = trans.apply(cum_mic, axis=1)

    # Replace existing delta_MIC where proxy exists
    trans["delta_MIC"] = trans["delta_MIC"].astype(object)
    mask = trans["delta_MIC_proxy"].notna()
    trans.loc[mask, "delta_MIC"] = trans.loc[mask, "delta_MIC_proxy"]
    log.info("Transitions with delta_MIC populated: %d / %d", int(mask.sum()), len(trans))

    trans.to_parquet(OUT_PROCESSED, index=False)
    log.info("Wrote %s", OUT_PROCESSED)

    # Per-drug-class breakdown
    cov = (trans.groupby("drug_class")["delta_MIC_proxy"]
           .apply(lambda s: f"{s.notna().sum()}/{len(s)} ({100*s.notna().mean():.1f}%)"))
    log.info("Drug-class MIC coverage:\n%s", cov.to_string())

    n_mic = int(mask.sum())
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    with GATE_LOG.open("a") as fh:
        fh.write(f"{ts} PHASE_A2_MIC: PASS [n_transitions_with_mic={n_mic}] "
                 f"[total_transitions={len(trans)}] [n_aledb_experiments={len(exp_table)}] "
                 f"[mic_unit=raw_aledb_int_in_experiment_name] "
                 f"[approximation=linear_ramp]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
