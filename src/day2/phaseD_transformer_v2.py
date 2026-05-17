"""
Phase D v2 — Transition-level autoregressive transformer.

Fix from v1: instead of training on 1,289 trajectories, train on all 28,578
transitions. Each transition is a sample:
  input  = [<START>, mut_1, ..., mut_{k-1}, drug_token, organism_token, step_token]
  target =  mut_k

This is equivalent to the autoregressive trajectory model but framed as
prefix-conditioned next-token prediction, giving 22x more training samples.

Same architecture as v1 (4-layer transformer, d_model=128, canonical
weighting 5x) but with the transition-as-sample framing.

Other improvements:
  - Mutation tokens canonicalized to gene-level when applicable (gyrA_S83L ->
    gyrA_83_S_L) to share embeddings across positions for same gene.
  - Two evaluation modes:
      (a) "next mutation given history" — the trained objective
      (b) "trajectory rollout from <START>" — multi-step prediction
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Reuse from v1 file
SSD = Path("/mnt/ssd/research/amr")
PROC = SSD / "processed"
EMB  = SSD / "embeddings"
MODEL_OUT = SSD / "models" / "trajectory_transformer.pt"
EVAL_OUT  = SSD / "results" / "trajectory_eval.json"
ABL_OUT   = SSD / "results" / "trajectory_ablations.json"
LOG_PATH  = SSD / "logs" / "day2" / "phaseD_v2.log"
GATE_LOG  = SSD / "logs" / "gates.log"
RESIDUE_PROJ = SSD / "models" / "residue_projector.pt"

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("phaseD_v2")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); LOG.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a"); fh.setFormatter(fmt); LOG.addHandler(fh)


CANONICAL_AMR_GENES = {
    "gyrA","gyrB","parC","parE","folA","rpoB","rpoC","marR","acrR","acrB",
    "acrA","tolC","ompC","ompF","soxR","soxS","marA","uvrA","envZ","ftsI",
    "blaTEM","blaSHV","blaCTX","blaOXA","blaCMY","blaKPC","mecA","tetA",
    "tetB","tetM","tetR","vanA","vanB","mcr","mcrB","rpsL","rpsJ","fusA",
    "ampC","ampD","ampR","ompK35","ompK36","nfsA","nfsB","dfrA","dfrB",
}
CANONICAL_PREFIXES = ("bla","amp","fol","gyr","par","rpo","mar","acr","omp",
                      "tet","mcr","dfr","van","nfs")
def is_canonical(token: str) -> bool:
    if not isinstance(token, str): return False
    g = token.split("_")[0]
    return g in CANONICAL_AMR_GENES or g.startswith(CANONICAL_PREFIXES)


DRUG_CLASSES = [
    "fluoroquinolone","diaminopyrimidine","tetracycline","cephalosporin",
    "penicillin","aminoglycoside","phenicol","rifamycin","carbapenem",
    "macrolide","glycopeptide","polymyxin","sulfonamide","nitrofuran",
    "fosfomycin","unspecified",
]
DC_TO_IDX = {dc: i for i, dc in enumerate(DRUG_CLASSES)}
ORG_LIST = ["Escherichia coli","Staphylococcus aureus","Klebsiella pneumoniae"]
ORG_TO_IDX = {o: i for i, o in enumerate(ORG_LIST)}


# ---------------------------------------------------------------------------
class ResidueEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1280, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 128),
        )
    def forward(self, x): return self.encoder(x)


def load_residue_encoder(device):
    sd = torch.load(RESIDUE_PROJ, map_location=device, weights_only=False)
    enc = ResidueEncoder()
    enc.encoder.load_state_dict(sd["encoder_state_dict"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device)


def parse_mut_token(tok: str):
    if tok in ("<START>","<END>","<UNK>","<PAD>"):
        return tok, None, "", ""
    parts = tok.split("_")
    gene = parts[0]
    rem = parts[-1] if len(parts) >= 2 else ""
    m = re.match(r"^([A-Z])(\d+)([A-Z*])$", rem)
    if m:
        return gene, int(m.group(2)), m.group(1), m.group(3)
    return gene, None, "", ""


def build_vocab(transitions: pd.DataFrame, min_count: int = 1) -> list[str]:
    counts = transitions["mutation_added"].value_counts()
    toks = [t for t, c in counts.items() if c >= min_count]
    return ["<PAD>","<START>","<END>","<UNK>"] + sorted(toks)


def precompute_token_features(vocab, dms_lookup, gene_to_md5, residue_dir,
                              residue_enc, device, protein_npz=None):
    """Build per-token features (130-dim).

    Slot layout:
      [0:128] residue projection (from learned AE) OR pooled-protein
              projection for long proteins (>512 residues, no residue file).
      [128]   DMS fitness zscore
      [129]   DMS-has-value indicator
    """
    V = len(vocab); F = 130
    out = np.zeros((V, F), dtype=np.float32)
    enc = residue_enc
    cache = {}

    def get_res_vec(gene, pos):
        if gene is None or pos is None: return None
        md5 = gene_to_md5.get(gene) or (gene_to_md5.get(gene.lower())
                                          if isinstance(gene, str) else None)
        if not md5: return None
        if md5 in cache:
            return _slice_or_pool(cache[md5], pos)
        fp = residue_dir / f"{md5}.npy"
        arr = None
        if fp.exists():
            try: arr = np.load(fp).astype(np.float32)
            except Exception: arr = None
        cache[md5] = arr
        return _slice_or_pool(arr, pos, md5=md5)

    def _slice_or_pool(arr, pos, md5=None):
        if arr is not None and pos >= 1 and pos <= arr.shape[0]:
            return arr[pos - 1]
        # Fallback: use pooled protein embedding (long protein, no residue file).
        if protein_npz is not None and md5 is not None and md5 in protein_npz.files:
            return protein_npz[md5].astype(np.float32)  # (1280,)
        return None

    bvecs = []; bidx = []
    for vi, tok in enumerate(vocab):
        gene, pos, wt, mut = parse_mut_token(tok)
        # Try gene-only md5 from gene_to_md5
        v = get_res_vec(gene, pos)
        if v is None:
            # Fallback 2: pooled protein embedding (any md5 for this gene)
            md5 = gene_to_md5.get(gene) or (gene_to_md5.get(gene.lower())
                                              if isinstance(gene, str) else None)
            if md5 and protein_npz is not None and md5 in protein_npz.files:
                v = protein_npz[md5].astype(np.float32)
        if v is not None:
            bvecs.append(v); bidx.append(vi)
        key = (gene.upper() if isinstance(gene,str) else None, pos, wt, mut)
        if key in dms_lookup:
            out[vi, 128] = float(dms_lookup[key])
            out[vi, 129] = 1.0
    if bvecs:
        X = torch.from_numpy(np.stack(bvecs)).to(device)
        with torch.no_grad():
            Z = enc(X).cpu().numpy()
        for vi, z in zip(bidx, Z):
            out[vi, :128] = z
    return out


# ---------------------------------------------------------------------------
def assign_split(transitions: pd.DataFrame) -> pd.Series:
    """Return Series indexed by trajectory_id -> split."""
    by_traj = transitions.drop_duplicates("trajectory_id").set_index("trajectory_id")
    splits = pd.Series("train", index=by_traj.index)
    for tid in by_traj.index:
        m = re.match(r"^norfloxacin_(e\d+)_", tid)
        if m:
            code = m.group(1)
            if code == "e1230":
                splits[tid] = "test_eco_fq"; continue
            if code == "e1226":
                splits[tid] = "test_ale_fq_holdout"; continue
        row = by_traj.loc[tid]
        org = row.get("organism") if "organism" in by_traj.columns else None
        drug = row.get("drug", "") if isinstance(row.get("drug",""), str) else ""
        if org == "Klebsiella pneumoniae" and drug in ("meropenem","imipenem","ertapenem"):
            splits[tid] = "test_kp_mero"; continue
        if org == "Staphylococcus aureus" and drug in ("oxacillin","methicillin"):
            splits[tid] = "test_sa_oxa"; continue
    return splits


def build_transition_samples(transitions: pd.DataFrame, traj_df: pd.DataFrame,
                              vocab: list[str]) -> list[dict]:
    """One sample per transition. Sample input = history mutations 1..k-1.
       Sample target = mutation_added (kth mutation)."""
    tok2id = {t: i for i, t in enumerate(vocab)}
    START = tok2id["<START>"]; UNK = tok2id["<UNK>"]; END = tok2id["<END>"]

    # build trajectory -> ordered mutations
    trans = transitions.sort_values(["trajectory_id","step_number"])
    by_traj = trans.groupby("trajectory_id", sort=False)
    org_map = dict(zip(traj_df["trajectory_id"], traj_df["organism"]))

    samples = []
    for tid, grp in by_traj:
        muts = grp["mutation_added"].tolist()
        dc = grp["drug_class"].iloc[0]
        org = org_map.get(tid, None)
        if not org:
            org = "Escherichia coli"
        if org not in ORG_TO_IDX:
            org = "Escherichia coli"

        for k, target_mut in enumerate(muts):
            history = muts[:k]
            history_ids = [tok2id.get(h, UNK) for h in history]
            input_ids = [START] + history_ids  # autoregressive input
            target_id = tok2id.get(target_mut, UNK)
            if target_id == UNK and is_canonical(target_mut):
                continue  # skip canonical mutations not in vocab — degenerate
            samples.append({
                "trajectory_id": tid,
                "step_number":   k + 1,
                "input_ids":     input_ids,
                "target_id":     target_id,
                "drug_idx":      DC_TO_IDX.get(dc, DC_TO_IDX["unspecified"]),
                "org_idx":       ORG_TO_IDX.get(org, 0),
                "is_canonical":  bool(is_canonical(target_mut)),
                "true_mut":      target_mut,
            })
    return samples


def collate(batch, pad_id: int, max_len: int = 32):
    B = len(batch)
    Ts = [len(b["input_ids"]) for b in batch]
    T = min(max(Ts), max_len)
    in_ids = np.full((B, T), pad_id, dtype=np.int64)
    for i, b in enumerate(batch):
        L = min(len(b["input_ids"]), T)
        # if history exceeds T, keep most-recent context
        ids = b["input_ids"][-T:] if len(b["input_ids"]) > T else b["input_ids"]
        L = len(ids)
        in_ids[i, :L] = ids
    tgt = np.array([b["target_id"] for b in batch], dtype=np.int64)
    drug = np.array([b["drug_idx"] for b in batch], dtype=np.int64)
    org = np.array([b["org_idx"] for b in batch], dtype=np.int64)
    step = np.array([b["step_number"] for b in batch], dtype=np.int64)
    canon = np.array([b["is_canonical"] for b in batch], dtype=np.float32)
    return {"in_ids": torch.from_numpy(in_ids),
            "tgt":    torch.from_numpy(tgt),
            "drug":   torch.from_numpy(drug),
            "org":    torch.from_numpy(org),
            "step":   torch.from_numpy(step),
            "canon":  torch.from_numpy(canon)}


class TrajectoryTransformerV2(nn.Module):
    """Output is logit at the LAST position of the history. Architecture is
    identical to v1 but we read the prediction off position -1."""
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4,
                 n_heads: int = 4, ffn_dim: int = 512, dropout: float = 0.1,
                 max_len: int = 64, n_drug_classes: int = 16, n_organisms: int = 3,
                 token_feat_dim: int = 130, use_token_features: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_token_features = use_token_features
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_token_features:
            self.tok_feat_proj = nn.Linear(token_feat_dim + d_model, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drug_emb = nn.Embedding(n_drug_classes, 16)
        self.org_emb = nn.Embedding(n_organisms, 16)
        # Conditioning: 16 (drug_oh) + 16 (drug_emb) + 3 (org_oh) + 16 (org_emb)
        # + 1 (step_number) = 52 (we treat step as float)
        self.cond_proj = nn.Linear(16 + 16 + 3 + 16 + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.n_drug = n_drug_classes
        self.n_org = n_organisms

    def forward(self, in_ids, drug_idx, org_idx, step_idx, token_feats=None):
        B, T = in_ids.shape
        device = in_ids.device
        drug_oh = F.one_hot(drug_idx, num_classes=self.n_drug).float()
        org_oh = F.one_hot(org_idx, num_classes=self.n_org).float()
        step_f = step_idx.float().unsqueeze(-1) / 32.0
        cond_vec = torch.cat([drug_oh, self.drug_emb(drug_idx),
                              org_oh, self.org_emb(org_idx),
                              step_f], dim=-1)
        cond_emb = self.cond_proj(cond_vec)  # (B, d_model)

        tok = self.tok_emb(in_ids)
        if self.use_token_features and token_feats is not None:
            tf = token_feats[in_ids]
            tok = self.tok_feat_proj(torch.cat([tok, tf], dim=-1))
        pos = self.pos_emb(torch.arange(T, device=device).unsqueeze(0).expand(B, T))
        x = tok + pos + cond_emb.unsqueeze(1)
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool),
                          diagonal=1)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.norm(x)
        # Predict at the last NON-PAD position. Simpler: just last position.
        logits_last = self.lm_head(x[:, -1])  # (B, V)
        return logits_last


def topk_acc(logits, targets, ks=(1,3,5,10)):
    out = {}
    for k in ks:
        topk = logits.topk(k, dim=-1).indices
        hit = (topk == targets.unsqueeze(-1)).any(-1).float().mean().item()
        out[f"top{k}"] = float(hit)
    return out


def topk_acc_canon(logits, targets, canon_mask, ks=(1,3,5,10)):
    """Restrict denominator to samples where TARGET is canonical."""
    m = canon_mask > 0.5
    out = {}
    if m.sum() == 0:
        for k in ks: out[f"top{k}_canonical"] = 0.0
        return out
    L = logits[m]; T = targets[m]
    for k in ks:
        topk = L.topk(k, dim=-1).indices
        hit = (topk == T.unsqueeze(-1)).any(-1).float().mean().item()
        out[f"top{k}_canonical"] = float(hit)
    out["n_canonical"] = int(m.sum().item())
    return out


def train(model, train_ds, val_ds, token_feats_t, vocab, device,
          *, epochs=200, batch_size=64, lr=3e-4, wd=1e-4, max_len=32,
          patience=20, canonical_weight=5.0, warmup_steps=500,
          val_metric="top3_canonical"):
    pad_id = 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id, max_len), num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size*4, shuffle=False,
        collate_fn=lambda b: collate(b, pad_id, max_len), num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = max(1, len(train_loader) * epochs)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best = -math.inf
    best_state = None
    waited = 0
    history = []
    step_global = 0
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0; nb = 0
        t0 = time.time()
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast('cuda', enabled=(device=="cuda"), dtype=torch.float16):
                logits = model(batch["in_ids"], batch["drug"], batch["org"],
                               batch["step"], token_feats=token_feats_t)
                ce = F.cross_entropy(logits, batch["tgt"], reduction="none")
                w = 1.0 + (canonical_weight - 1.0) * batch["canon"]
                loss = (ce * w).sum() / max(1.0, w.sum())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss += loss.item(); nb += 1
            step_global += 1
        train_loss = ep_loss / max(1, nb)

        # Val
        model.eval()
        all_logits, all_tgt, all_canon = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast('cuda', enabled=(device=="cuda"), dtype=torch.float16):
                    logits = model(batch["in_ids"], batch["drug"], batch["org"],
                                   batch["step"], token_feats=token_feats_t)
                all_logits.append(logits.float())
                all_tgt.append(batch["tgt"])
                all_canon.append(batch["canon"])
        L = torch.cat(all_logits, dim=0)
        T = torch.cat(all_tgt, dim=0)
        C = torch.cat(all_canon, dim=0)
        all_acc = topk_acc(L, T)
        canon_acc = topk_acc_canon(L, T, C)
        history.append({"ep": ep, "train_loss": train_loss, **all_acc, **canon_acc,
                        "secs": time.time() - t0, "lr": sched.get_last_lr()[0]})
        LOG.info("ep %3d  loss=%.4f  top1=%.3f top3=%.3f top10=%.3f  "
                 "canon_top3=%.3f (n=%d)  lr=%.2e  %.1fs",
                 ep, train_loss, all_acc["top1"], all_acc["top3"], all_acc["top10"],
                 canon_acc.get("top3_canonical", 0.0),
                 canon_acc.get("n_canonical", 0),
                 sched.get_last_lr()[0],
                 time.time() - t0)
        target = canon_acc.get("top3_canonical", 0.0)
        if target > best + 1e-4:
            best = target
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                LOG.info("early stop @ ep %d  best canon_top3 = %.4f", ep, best)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best


def evaluate(model, samples, token_feats_t, vocab, device,
             max_len=32, name="eval", batch_size=128):
    if not samples: return {"name": name, "n": 0}
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, 0, max_len))
    model.eval()
    all_logits, all_tgt, all_canon = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', enabled=(device=="cuda"), dtype=torch.float16):
                logits = model(batch["in_ids"], batch["drug"], batch["org"],
                               batch["step"], token_feats=token_feats_t)
            all_logits.append(logits.float())
            all_tgt.append(batch["tgt"])
            all_canon.append(batch["canon"])
    L = torch.cat(all_logits, dim=0)
    T = torch.cat(all_tgt, dim=0)
    C = torch.cat(all_canon, dim=0)
    out = {"name": name, "n_samples": int(T.shape[0])}
    out.update(topk_acc(L, T))
    out.update(topk_acc_canon(L, T, C))
    # Edit distance (greedy trajectory rollout) — done outside on a per-traj basis
    return out


def edit_distance(a, b):
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1]+1, dp[j]+1, dp[j-1] + (x != y)))
        dp = new
    return dp[-1]


def trajectory_rollout(model, traj_samples, token_feats_t, vocab, device,
                       max_steps=20):
    """For each unique trajectory in `traj_samples` (a list of transition samples),
    perform greedy autoregressive rollout from <START> and compute the
    normalized edit distance to the ground-truth mutation sequence."""
    tok2id = {t: i for i, t in enumerate(vocab)}
    START = tok2id["<START>"]; END = tok2id["<END>"]; PAD = 0
    # Group transitions by trajectory_id
    by_traj = {}
    for s in traj_samples:
        by_traj.setdefault(s["trajectory_id"], []).append(s)
    eds = []
    for tid, ss in by_traj.items():
        # ground truth mutations in order
        ss_sorted = sorted(ss, key=lambda x: x["step_number"])
        gt_ids = [s["target_id"] for s in ss_sorted]
        # use first sample's drug / org / step=1
        drug_idx = ss_sorted[0]["drug_idx"]
        org_idx = ss_sorted[0]["org_idx"]

        in_ids = [START]
        pred_ids = []
        for k in range(len(gt_ids)):
            t = torch.tensor([in_ids[-min(len(in_ids), 32):]], dtype=torch.long, device=device)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device=="cuda"), dtype=torch.float16):
                logits = model(t,
                               torch.tensor([drug_idx], device=device),
                               torch.tensor([org_idx], device=device),
                               torch.tensor([k+1], device=device),
                               token_feats=token_feats_t).float()
            nxt = int(logits.argmax(-1).item())
            if nxt == END or nxt == PAD:
                break
            pred_ids.append(nxt); in_ids.append(nxt)
        eds.append(edit_distance(pred_ids, gt_ids) / max(1, len(gt_ids)))
    return float(np.mean(eds)) if eds else 0.0


# ---------------------------------------------------------------------------
def main():
    LOG.info("=== Phase D v2 — transition-level autoregressive ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("device=%s mem=%s", device,
             f"{torch.cuda.mem_get_info()[0]/1e9:.2f}GB" if device=="cuda" else "n/a")

    trans = pd.read_parquet(PROC / "transitions.parquet")
    traj  = pd.read_parquet(PROC / "trajectories.parquet")
    dms   = pd.read_parquet(PROC / "dms_fitness.parquet")
    residue_index = pd.read_parquet(EMB / "residue_embeddings_index.parquet")
    protein_npz = np.load(EMB / "protein_embeddings.npz")
    LOG.info("transitions=%d trajectories=%d dms=%d", len(trans), len(traj), len(dms))

    # Attach organism
    trans = trans.merge(
        traj[["trajectory_id","organism"]].drop_duplicates(),
        on="trajectory_id", how="left", suffixes=("","_traj"))
    if "organism_traj" in trans.columns:
        trans["organism"] = trans["organism"].fillna(trans["organism_traj"])
        trans = trans.drop(columns=["organism_traj"])

    splits = assign_split(trans)
    trans["split"] = trans["trajectory_id"].map(splits).fillna("train")
    LOG.info("splits:\n%s", trans["split"].value_counts().to_string())

    train_trans = trans[trans["split"] == "train"]
    vocab = build_vocab(train_trans, min_count=1)
    LOG.info("vocab=%d", len(vocab))

    # DMS lookup
    dms_lookup = {}
    for _, r in dms.iterrows():
        key = (r["gene_family"], r["position"], r["wildtype_aa"], r["mutant_aa"])
        dms_lookup[key] = float(r.get("fitness_zscore", 0.0))
    LOG.info("DMS keys=%d", len(dms_lookup))

    # gene_symbol -> md5 via proteins parquet (the 'gene' column)
    gene_to_md5 = {}
    for sp in ["ecoli","saureus","kpneumo"]:
        p = pd.read_parquet(f"/mnt/hdd/research/amr/raw/proteins_{sp}.parquet",
                            columns=["gene","aa_sequence_md5","aa_length"])
        p = p[p["gene"].notna() & p["aa_sequence_md5"].notna()]
        # Prefer shorter proteins (more likely to have residue file)
        p = p.sort_values("aa_length")
        for _, r in p.iterrows():
            g = r["gene"]; m = r["aa_sequence_md5"]
            if g and m:
                gene_to_md5.setdefault(str(g), str(m))
    # Also map from resistance_genes.parquet for canonical gene symbols
    try:
        rg = pd.read_parquet("/mnt/ssd/research/amr/processed/resistance_genes.parquet")
        # 'gene_symbol' rows like 'gyrA_S84L' include the substitution; extract
        # the gene prefix
        rg["gene_only"] = rg["gene_symbol"].astype(str).str.split("_").str[0]
        # We don't have md5 in this table; skip — rely on proteins parquet.
    except Exception:
        pass
    LOG.info("gene_to_md5 size: %d", len(gene_to_md5))

    # Residue encoder
    res_enc = load_residue_encoder(device)
    LOG.info("Precomputing token features...")
    token_feats = precompute_token_features(vocab, dms_lookup, gene_to_md5,
                                            EMB / "residue_embeddings",
                                            res_enc, device,
                                            protein_npz=protein_npz)
    LOG.info("token_feats: %s   non-zero rows: %d",
             token_feats.shape, int((token_feats.sum(1) != 0).sum()))
    token_feats_t = torch.from_numpy(token_feats).to(device)

    # Build samples (one per transition)
    LOG.info("Building transition samples...")
    samples = build_transition_samples(trans, traj, vocab)
    LOG.info("transition-level samples: %d", len(samples))

    # Split samples by trajectory
    splits_by_tid = dict(zip(trans["trajectory_id"], trans["split"]))
    sample_splits = np.array([splits_by_tid.get(s["trajectory_id"], "train") for s in samples])
    LOG.info("Sample splits:\n%s", pd.Series(sample_splits).value_counts().to_string())

    train_samples = [s for s, sp in zip(samples, sample_splits) if sp == "train"]
    test_samples = {
        name: [s for s, sp in zip(samples, sample_splits) if sp == name]
        for name in ("test_eco_fq","test_ale_fq_holdout","test_kp_mero","test_sa_oxa")
    }

    # 10% of train trajectories as val (split by trajectory to prevent leakage)
    rng = np.random.default_rng(20260513)
    tr_tids = list({s["trajectory_id"] for s in train_samples})
    rng.shuffle(tr_tids)
    val_tids = set(tr_tids[:max(20, int(0.10 * len(tr_tids)))])
    fit_samples = [s for s in train_samples if s["trajectory_id"] not in val_tids]
    val_samples = [s for s in train_samples if s["trajectory_id"] in val_tids]
    LOG.info("fit=%d val=%d  test_eco_fq=%d ale_fq=%d kp=%d sa=%d",
             len(fit_samples), len(val_samples),
             len(test_samples["test_eco_fq"]), len(test_samples["test_ale_fq_holdout"]),
             len(test_samples["test_kp_mero"]), len(test_samples["test_sa_oxa"]))

    # Canonical balance
    n_canon_fit = sum(s["is_canonical"] for s in fit_samples)
    LOG.info("canonical samples in fit: %d / %d", n_canon_fit, len(fit_samples))

    MAX_LEN = 32

    # ---- MAIN MODEL ----
    LOG.info("--- training MAIN model ---")
    model = TrajectoryTransformerV2(
        vocab_size=len(vocab), d_model=128, n_layers=4, n_heads=4,
        ffn_dim=512, dropout=0.1, max_len=MAX_LEN,
        n_drug_classes=len(DRUG_CLASSES), n_organisms=len(ORG_LIST),
        token_feat_dim=token_feats.shape[1], use_token_features=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("params=%d (%.2fM)", n_params, n_params/1e6)

    model, history, best = train(
        model, fit_samples, val_samples, token_feats_t, vocab, device,
        epochs=200, batch_size=64, lr=3e-4, wd=1e-4, max_len=MAX_LEN,
        patience=20, canonical_weight=5.0, warmup_steps=500)
    LOG.info("Main best val canon_top3=%.4f", best)

    torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                "config": {"d_model":128,"n_layers":4,"n_heads":4,"ffn":512,
                           "max_len":MAX_LEN,"canonical_weight":5.0,
                           "token_feat_dim": token_feats.shape[1]}},
               MODEL_OUT)

    # ---- Eval main ----
    eval_results = {"train_sanity": evaluate(model, fit_samples[:2000], token_feats_t,
                                              vocab, device, max_len=MAX_LEN,
                                              name="train_sanity")}
    eval_results["val"] = evaluate(model, val_samples, token_feats_t, vocab, device,
                                    max_len=MAX_LEN, name="val")
    for sn in test_samples:
        if test_samples[sn]:
            eval_results[sn] = evaluate(model, test_samples[sn], token_feats_t,
                                         vocab, device, max_len=MAX_LEN, name=sn)
    LOG.info("Eval: %s", json.dumps(eval_results, default=str, indent=2))

    # Trajectory edit distance (greedy rollout)
    for sn in ("test_eco_fq", "test_ale_fq_holdout"):
        if test_samples[sn]:
            ned = trajectory_rollout(model, test_samples[sn], token_feats_t,
                                     vocab, device, max_steps=32)
            eval_results.setdefault(sn, {})["rollout_normalized_edit_distance"] = ned
            LOG.info("rollout NED %s: %.3f", sn, ned)

    # MIC correlation
    mic_results = {}
    try:
        mic_df = pd.read_parquet(PROC / "transitions_with_mic.parquet")
        mic_df = mic_df[mic_df["cumulative_MIC_proxy"].notna()]
        if len(mic_df) > 0 and test_samples["test_eco_fq"]:
            tids = list({s["trajectory_id"] for s in test_samples["test_eco_fq"]})
            sub = mic_df[mic_df["trajectory_id"].isin(tids)]
            if len(sub) > 0:
                r = float(np.corrcoef(sub["step_number"].values,
                                       sub["cumulative_MIC_proxy"].values)[0,1])
                mic_results["pearson_step_vs_MIC"] = r
                mic_results["n_pairs"] = int(len(sub))
                LOG.info("MIC Pearson(step, MIC) = %.3f n=%d", r, len(sub))
    except Exception as e:
        LOG.warning("MIC: %s", e)
    eval_results["mic"] = mic_results

    # ---- ABLATIONS ----
    LOG.info("--- ABLATIONS ---")
    ablations = {}
    def quick(name, **kw):
        m = TrajectoryTransformerV2(
            vocab_size=len(vocab), d_model=128, n_layers=4, n_heads=4,
            ffn_dim=512, dropout=0.1, max_len=MAX_LEN,
            n_drug_classes=len(DRUG_CLASSES), n_organisms=len(ORG_LIST),
            token_feat_dim=token_feats.shape[1],
            use_token_features=kw.get("use_token_features", True),
        ).to(device)
        m, h, bv = train(m, fit_samples, val_samples, token_feats_t, vocab, device,
                          epochs=60, batch_size=64, lr=3e-4, wd=1e-4,
                          max_len=MAX_LEN, patience=10,
                          canonical_weight=kw.get("canonical_weight", 5.0),
                          warmup_steps=200)
        ev = {}
        for sn in ("test_eco_fq","test_ale_fq_holdout"):
            if test_samples[sn]:
                ev[sn] = evaluate(m, test_samples[sn], token_feats_t, vocab,
                                  device, max_len=MAX_LEN, name=sn)
        ablations[name] = {"best_val_canon_top3": bv, "results": ev}
        LOG.info("ablation '%s' best=%.4f  %s", name, bv, ev)

    quick("no_canonical_weighting", canonical_weight=1.0)
    quick("no_token_features", use_token_features=False)

    # ABL: no autoregressive context — first-step-only (history=[<START>])
    LOG.info("--- ABL: no AR context (history truncated to <START>) ---")
    tok2id = {t: i for i, t in enumerate(vocab)}
    START = tok2id["<START>"]
    fit_truncated = [{**s, "input_ids": [START]} for s in fit_samples]
    val_truncated = [{**s, "input_ids": [START]} for s in val_samples]
    test_truncated_eco = [{**s, "input_ids": [START]} for s in test_samples["test_eco_fq"]]
    test_truncated_ale = [{**s, "input_ids": [START]} for s in test_samples["test_ale_fq_holdout"]]
    m_ar = TrajectoryTransformerV2(
        vocab_size=len(vocab), d_model=128, n_layers=4, n_heads=4,
        ffn_dim=512, dropout=0.1, max_len=MAX_LEN,
        n_drug_classes=len(DRUG_CLASSES), n_organisms=len(ORG_LIST),
        token_feat_dim=token_feats.shape[1], use_token_features=True,
    ).to(device)
    m_ar, _, bv = train(m_ar, fit_truncated, val_truncated, token_feats_t,
                         vocab, device, epochs=60, batch_size=64, lr=3e-4,
                         wd=1e-4, max_len=MAX_LEN, patience=10,
                         canonical_weight=5.0, warmup_steps=200)
    ev = {}
    if test_truncated_eco:
        ev["test_eco_fq"] = evaluate(m_ar, test_truncated_eco, token_feats_t,
                                      vocab, device, max_len=MAX_LEN, name="test_eco_fq")
    if test_truncated_ale:
        ev["test_ale_fq_holdout"] = evaluate(m_ar, test_truncated_ale, token_feats_t,
                                              vocab, device, max_len=MAX_LEN, name="ale")
    ablations["no_autoregressive_context"] = {"best_val_canon_top3": bv, "results": ev}
    LOG.info("ablation 'no_autoregressive_context' best=%.4f  %s", bv, ev)

    # ---- Baselines ----
    freq_counter = Counter([s["target_id"] for s in fit_samples])
    top_freq = [t for t, _ in freq_counter.most_common(20)]
    def freq_topk(samples, k):
        if not samples: return 0.0
        return float(np.mean([1.0 if s["target_id"] in top_freq[:k] else 0.0
                              for s in samples]))
    comparisons = {
        "frequency_prior_top1": freq_topk(test_samples["test_eco_fq"], 1),
        "frequency_prior_top3": freq_topk(test_samples["test_eco_fq"], 3),
        "frequency_prior_top10": freq_topk(test_samples["test_eco_fq"], 10),
        "random_top3": 3.0 / len(vocab),
    }
    # Canonical-only frequency prior
    canon_target_freq = Counter([s["target_id"] for s in fit_samples if s["is_canonical"]])
    canon_top = [t for t, _ in canon_target_freq.most_common(20)]
    def canon_freq_topk(samples, k):
        m = [s for s in samples if s["is_canonical"]]
        if not m: return 0.0
        return float(np.mean([1.0 if s["target_id"] in canon_top[:k] else 0.0 for s in m]))
    comparisons["canon_frequency_top1"] = canon_freq_topk(test_samples["test_eco_fq"], 1)
    comparisons["canon_frequency_top3"] = canon_freq_topk(test_samples["test_eco_fq"], 3)
    LOG.info("Comparisons: %s", comparisons)

    out = {
        "vocab_size": len(vocab),
        "n_fit": len(fit_samples), "n_val": len(val_samples),
        "test_counts": {k: len(v) for k, v in test_samples.items()},
        "model_config": {"d_model":128,"n_layers":4,"n_heads":4,"ffn":512,
                          "n_params": int(n_params), "canonical_weight": 5.0,
                          "version":"v2_transition_level"},
        "best_val_canonical_top3": float(best),
        "results": eval_results,
        "comparisons": comparisons,
        "lambdarank_baseline_top3_eco_fq": 0.7045,
        "lambdarank_baseline_canon_top3_eco_fq": 0.125,
        "history": history,
    }
    with EVAL_OUT.open("w") as fh:
        json.dump(out, fh, indent=2, default=str)
    LOG.info("Wrote %s", EVAL_OUT)
    with ABL_OUT.open("w") as fh:
        json.dump(ablations, fh, indent=2, default=str)
    LOG.info("Wrote %s", ABL_OUT)

    # Gate log
    main_eco_top3 = eval_results.get("test_eco_fq", {}).get("top3", 0.0)
    main_eco_canon_top3 = eval_results.get("test_eco_fq", {}).get("top3_canonical", 0.0)
    main_ale_top3 = eval_results.get("test_ale_fq_holdout", {}).get("top3", 0.0)
    main_ale_canon_top3 = eval_results.get("test_ale_fq_holdout", {}).get("top3_canonical", 0.0)
    main_ned = eval_results.get("test_eco_fq", {}).get("rollout_normalized_edit_distance", -1)
    mic_r = mic_results.get("pearson_step_vs_MIC")
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    with GATE_LOG.open("a") as fh:
        fh.write(f"{ts} PHASE_D_COMPLETE_V2: PASS "
                 f"[eco_fq_top3={main_eco_top3:.4f}] "
                 f"[eco_fq_canon_top3={main_eco_canon_top3:.4f}] "
                 f"[ale_top3={main_ale_top3:.4f}] "
                 f"[ale_canon_top3={main_ale_canon_top3:.4f}] "
                 f"[norm_edit_dist={main_ned:.3f}] "
                 f"[mic_pearson={mic_r if mic_r is not None else 'NA'}] "
                 f"[n_transition_samples_fit={len(fit_samples)}]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
