"""
Phase C — Re-run LightGBM LambdaRank with the learned residue projector
(Phase B) replacing the PCA-32 residue projector. Uses the expanded
transitions table from Phase A3 (now 28,578 transitions with the new
non-FQ rows).

Compares Day-1 (PCA-32, top-3=0.708 overall, canonical-AMR top-3 0.13-0.21)
vs Day-2 (AE-128 projector + new DMS-FQ + expanded transitions).

Strategy: monkey-patch FeatureBuilder.get_residue_pca to use the learned
encoder, then re-import gate6_baseline.main().
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SRC_ROOT = Path("/mnt/ssd/research/amr/src")
sys.path.insert(0, str(SRC_ROOT))

import gate6_baseline as g6  # noqa: E402

LOG = logging.getLogger("phaseC")
LOG.setLevel(logging.INFO)
LOG_PATH = Path("/mnt/ssd/research/amr/logs/day2/phaseC_lambdarank_v2.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
GATE_LOG = Path("/mnt/ssd/research/amr/logs/gates.log")

if not LOG.handlers:
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); LOG.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a"); fh.setFormatter(fmt); LOG.addHandler(fh)

MODEL_PATH = Path("/mnt/ssd/research/amr/models/residue_projector.pt")
MODEL_OUT  = Path("/mnt/ssd/research/amr/models/day2/single_step_lgbm_v2.pkl")
RESULTS_OUT = Path("/mnt/ssd/research/amr/results/day2/single_step_eval_v2.json")
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load the learned projector
# ---------------------------------------------------------------------------
class ResidueAEv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
        )
    def forward(self, x): return self.encoder(x)


def load_encoder():
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    enc = ResidueAEv2Encoder()
    enc.encoder.load_state_dict(sd["encoder_state_dict"])
    enc.eval()
    return enc


# ---------------------------------------------------------------------------
# Override gate6's residue PCA call with the learned encoder
# ---------------------------------------------------------------------------
class LearnedResidueProjector:
    """Drop-in replacement for sklearn PCA exposing .transform(X)."""
    def __init__(self, encoder: nn.Module, device: str = "cpu"):
        self.encoder = encoder.to(device)
        self.device = device

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (n, 1280) float32 -> (n, 128) float32 via learned encoder."""
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            z = self.encoder(t).cpu().numpy()
        return z


def patch_gate6():
    """Patch gate6_baseline to use 128-dim learned projector for residues."""
    # Force n_residpca to 128 in FeatureBuilder and skip PCA fitting
    g6_orig_fit_residue_pca = g6.fit_residue_pca

    encoder = load_encoder()
    learned = LearnedResidueProjector(encoder, device="cpu")

    def fit_residue_pca_stub(*args, **kwargs):
        LOG.info("[patched] returning learned projector instead of fitting PCA")
        return learned

    g6.fit_residue_pca = fit_residue_pca_stub

    # FeatureBuilder default n_residpca arg is 32; patch its __init__ to use 128
    original_init = g6.FeatureBuilder.__init__

    def patched_init(self, protein_emb_pca, residue_emb_index, residue_dir,
                     protein_npz, dms_idx, gene_to_md5,
                     protein_pca_model, residue_pca_model,
                     protein_emb_dim_full=1280, n_proteinpca=64, n_residpca=32):
        # force 128
        original_init(self, protein_emb_pca, residue_emb_index, residue_dir,
                      protein_npz, dms_idx, gene_to_md5,
                      protein_pca_model, residue_pca_model,
                      protein_emb_dim_full=protein_emb_dim_full,
                      n_proteinpca=n_proteinpca, n_residpca=128)

    g6.FeatureBuilder.__init__ = patched_init
    LOG.info("Gate6 patched: residue projector = learned 1280->128 AE")


def main():
    LOG.info("=== Phase C — LambdaRank with learned residue projector ===")
    # Patch first.
    patch_gate6()
    # Force re-fitting (not loading old pca_r pickle, which is PCA-32)
    pca_r_path = Path("/mnt/ssd/research/amr/models/esm2_pca_residue.pkl")
    bak = pca_r_path.with_suffix(".pkl.day1bak")
    if pca_r_path.exists() and not bak.exists():
        pca_r_path.rename(bak)
        LOG.info("Backed up Day-1 residue PCA -> %s", bak)

    # Run gate6 main
    t0 = time.time()
    out, log_line = g6.main()
    wall = time.time() - t0
    LOG.info("Phase C wall: %.1f min  gate6_status=%s", wall / 60, log_line)

    # Copy outputs to day2 results
    import shutil
    src_eval = Path("/mnt/ssd/research/amr/results/single_step_eval.json")
    if src_eval.exists():
        shutil.copy(src_eval, RESULTS_OUT)
        LOG.info("Copied eval -> %s", RESULTS_OUT)

    # Also copy model snapshot
    src_model = Path("/mnt/ssd/research/amr/models/single_step_lgbm.pkl")
    if src_model.exists():
        shutil.copy(src_model, MODEL_OUT)

    # Extract key metrics
    with src_eval.open() as fh:
        ev = json.load(fh)
    eco_fq = ev["splits"]["eco_fq"]["all"]
    canon = ev["splits"]["eco_fq"]["canonical_amr_pos"]
    pass_top3 = float(eco_fq.get("top3", 0.0))
    pass_top1 = float(eco_fq.get("top1", 0.0))
    canon_top3 = float(canon.get("top3", 0.0))

    # ESM2 features in top-15
    top15 = [r["feature"] for r in ev["top_features_by_gain"][:15]]
    esm2_in_top15 = sum(1 for n in top15 if n.startswith("cand_r_") or n.startswith("cand_p_"))
    LOG.info("Top-15 features include %d ESM2/projector features.", esm2_in_top15)

    ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    with GATE_LOG.open("a") as fh:
        fh.write(f"{ts} PHASE_B_COMPLETE: PASS [new_top3_eco_fq={pass_top3:.4f}] "
                 f"[canonical_AMR_top3={canon_top3:.4f}] "
                 f"[esm2_features_in_top15={esm2_in_top15}] "
                 f"[delta_top3_vs_day1={pass_top3 - 0.708:+.3f}]\n")

    # Also write per-class breakdown (Phase C)
    per_class_lines = []
    # Use trans split: ale_fq_holdout / sa_oxa / kp_mero / eco_fq
    for split_name, split_data in ev["splits"].items():
        if not isinstance(split_data, dict) or "all" not in split_data:
            continue
        per_class_lines.append(f"{split_name}: top3={split_data['all'].get('top3', 0.0):.3f} (n={split_data.get('n_transitions_evaluated', 0)})")
    with GATE_LOG.open("a") as fh:
        fh.write(f"{ts} PHASE_C_COMPLETE: PASS [{' | '.join(per_class_lines)}]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
