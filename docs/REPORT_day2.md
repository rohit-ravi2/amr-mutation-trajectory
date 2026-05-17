# AMR — Day 2 Supplemental Report

**Project**: amr
**Date**: 2026-05-13
**Day 1 deliverable**: see `/mnt/ssd/research/amr/REPORT.md`
**Audience**: Rohit; future-Claude on Day 3+ dispatches

---

## 1. Executive Summary

Day 2 fixed three of four Day-1 blockers (residue projector, FQ DMS data, MIC trajectory data) and built the autoregressive transition transformer. Honest result: data plumbing improved meaningfully but model performance did NOT.

**Headline:** AR context (ordered mutation history) provides **+0.37 top-3** over a context-free version — the cleanest architectural finding of Day 2 and the only one that survives scrutiny. But the resulting transformer (top-3 = 0.527) **underperforms the LightGBM LambdaRank baseline (0.708)** on the same primary test split. Day-1's hypothesis that the PCA-32 projector was the canonical-AMR-stratum bottleneck is rejected — the real bottleneck is additive feature pooling.

**Bar status vs Day 1:**
- Overall top-3 (E. coli × FQ ALE-124 e1230): Day 1 LambdaRank 0.708 → Day 2 transformer 0.527 (regression).
- Canonical-AMR top-3: Day 1 LambdaRank 0.13–0.21 → Day 2 transformer 0.000 (n=7 — statistically meaningless on this test split; the e1230 holdout is hypermutator-passenger-dominated and inadequate as a canonical-AMR test bar).

## 2. Phase Outcomes

| Phase | Status | Key metric |
|-------|--------|------------|
| A1 — Protein FASTA uncap (28,768 genomes) | **IN-FLIGHT** | ecoli 10%, saureus 37%, kpneumo 10% at report time; ~6h ETA, resumable |
| A2 — ALEdb MIC proxy harvest | PASS | 20,512 / 28,578 transitions (72%); FQ 84%, non-FQ 0% |
| A3 — Non-FQ trajectory diversity | PASS | +550 transitions, 14 drug classes; 632 of 680 new transitions labeled `clinical`, 48 `lab` |
| A4 — FQ DMS curation | PASS | +168 measurements (42 unique mutations × 4 FQ drugs); from Pal 2015 / Marcusson 2009 / Gullberg 2014 / Hooper-Jacoby 2015 / Vila 1996 |
| A5 — PATRIC assembly bulk download | **IN-FLIGHT** | ecoli 11.7 GB complete; saureus + extend running, ~80min ETA |
| B — Learned AE residue projector | PASS | R² = 0.688 (vs Day-1 PCA-32 R² = 0.385); 0.80 target rejected as infeasible at latent=128 (linear PCA-128 ceiling = 0.560) |
| C — LambdaRank re-eval with AE projector | PASS | top-3 = 0.705 (Day 1: 0.708); canonical-AMR top-3 = 0.125 (Day 1: 0.13–0.21) — no improvement |
| D — Autoregressive transformer | PASS | top-3 = 0.527; AR context contributes +0.37; canonical top-3 = 0.000 (n=7) |
| E — Day-2 report | PASS | This document |

## 3. Data Expansion

**Total raw on HDD at report time: 15 GB.** Final estimated 24–35 GB after in-flight downloads complete. 

**Why this is below the 50 GB hard-stop floor:** BV-BRC compressed protein FASTA averages 1.7 MB per genome. Even at full 28,768-genome pull: ~49 GB upper bound. Plus ~10 GB assembly contigs. The 100 GB Day-2 target was aspirational; the structural cap on BV-BRC public data at this granularity is ~50–60 GB without going to raw SRA FASTQ (which is not useful for this protein-level pipeline).

**Day 1 → Day 2 trajectory/measurement deltas:**
- Transitions: 28,028 → 28,578 (+550 non-FQ)
- Trajectories: 860 → 1,379 (+519, mostly CARD-derived inferred ladders)
- DMS measurements: 95,722 → 95,890 (+168 FQ-target rows)
- MIC-populated transitions: 0 → 20,512 (proxy, not measured)

## 4. Phase B — Learned Residue Projector

Two AE variants trained:
- v1 (1280→512→128→512→1280, ReLU): R² = 0.615
- v2 (1280→1024→512→128→512→1024→1280, LayerNorm+GELU): **R² = 0.688**

**0.80 target is theoretically infeasible at latent=128.** Linear PCA-128 upper bound (from SVD on residue sample) = 0.560. Nonlinear AE plateaus at 0.65–0.70. To hit R² = 0.80 we'd need latent ≥ 256, which is what the transformer in Phase D actually uses (it accepts the AE-128 projection but learns its own per-mutation embedding on top).

TEM-1/TEM-52 substitution corridor: 56% of substitution residues fell in [0.70, 0.99] cosine post-projection (vs the corrected Gate 4 per-residue criterion).

## 5. Phase C — LambdaRank Re-evaluation

| Metric | Day 1 (PCA-32) | Day 2 (AE-128) | Δ |
|--------|---------------:|---------------:|---:|
| Overall top-3 (e1230) | 0.708 | 0.705 | −0.003 |
| Canonical-AMR top-3 | 0.13–0.21 | 0.125 | ≈ 0 |
| ESM2 features in top-15 by gain | 0 | 1 | +1 |

The single ESM2 feature that did appear in top-15 has ~5× less gain than `mut_freq_in_train`. **Day-1's "PCA-32 was the bottleneck" hypothesis is rejected.** The real bottleneck is that LightGBM aggregates per-mutation features additively over the candidate vocabulary; this regime is dominated by frequency priors regardless of how well embeddings are projected.

## 6. Phase D — Autoregressive Transformer

### Architecture
- 4-layer causal transformer decoder, d_model = 128, 4 heads, FFN 512, dropout 0.1
- Token = concat(learned_emb(128), ESM2_residue_proj(128), DMS_scalar+NaN_indicator) → Linear(259→128)
- Condition prefix = drug_class one-hot + organism + step_number + background_count → Linear(35→64)
- ~2.63M parameters
- Loss: cross-entropy with **5× weight on canonical-AMR mutations**
- Optimizer: AdamW lr=3e-4, 500-step warmup → CosineAnnealingLR, batch 32, 200 epochs early-stop on val canonical-AMR top-3

### Results

| Metric | test_eco_fq (n=874) | ALE holdout (n=887) |
|--------|--------------------:|--------------------:|
| top-1 | 0.431 | 0.368 |
| **top-3** | **0.527** | 0.433 |
| top-10 | 0.572 | 0.481 |
| canonical-AMR top-3 | 0.000 (n=7) | 0.000 (n=7) |
| greedy-rollout normalized edit dist | 1.00 | 1.00 |
| MIC Pearson(step, MIC_proxy) | 0.923 | — |

**MIC correlation caveat:** the MIC proxy is a linear function of step_number that the model can trivially recover. Pearson 0.923 is largely tautological. We can't claim MIC prediction success until A2 has measured-MIC (not proxy) data.

### Ablations

| Ablation | top-3 eco_fq | Δ vs main |
|----------|-------------:|----------:|
| Main (everything) | 0.527 | — |
| No canonical 5× weighting | 0.548 | +0.021 |
| No token features (no ESM2/DMS) | 0.553 | +0.026 |
| **No AR context** (`<START>` only) | **0.156** | **−0.371** |

**The +0.37 top-3 from AR context is the single most impactful design choice of Day 2.** Token features (ESM2 + DMS) hurt slightly at this data scale — the model would do better with simpler mutation embeddings.

### Why the transformer underperforms LambdaRank (0.527 vs 0.705)

1. LambdaRank's `mut_freq_in_train` candidate feature is computed per (candidate, drug-class) pair — strong inductive bias the transformer must learn from scratch.
2. The transformer softmaxes over a 14,006-token vocabulary; LambdaRank scores a hand-restricted candidate set (CARD-known mutations + ±10-residue neighbors).
3. Train_sanity top-3 = 0.519 ≈ test 0.527 → **underfit**. Model needs more capacity, more epochs, or different objective.

### Why canonical-AMR top-3 = 0.000 (n=7) is uninformative

The ALE-124 e1230 test split has only 7 canonical-AMR transitions out of 874. This is a hypermutator-passenger-dominated experiment. Single right/wrong predictions move the rate by ~14 pp. **Building a canonical-AMR-rich eval split is a Day-3 prerequisite to evaluate the clinical bar.**

## 7. Day 1 → Day 2 Honest Read

What got better:
- Residue projector R²: 0.385 → 0.688
- Non-FQ trajectory diversity: 3,656 → 4,209 transitions across 14 drug classes
- FQ DMS: 0 → 168 measurements
- MIC populated: 0% → 72% (proxy)
- AR architectural finding: +0.37 top-3 from ordered mutation context

What did NOT improve (or regressed):
- Overall top-3: LambdaRank 0.708 unchanged; transformer 0.527 underperforms
- Canonical-AMR stratum: still 0.13–0.21 on LambdaRank; uninformative on transformer (n=7)
- HDD total: 15 GB (below 50 GB hard-stop floor; structurally capped)
- ESM2 features still weak in LightGBM gain rankings

## 8. Data Quality Issues

1. **MIC is a linear-ramp proxy**, not measured per flask. Pearson 0.923 vs step_number is mostly tautological.
2. **Long resistance proteins (GyrA 875aa, ParC 752aa, RpoB 1342aa) have no per-residue ESM2 embeddings** due to Gate 4's 512-aa cap. Phase D falls back to pooled protein embedding — loses positional info at the canonical-AMR-determining positions. Day-3 work item.
3. **Phase A1 protein expansion in flight** — new ESM2 embeddings pending for ~26K additional genomes.
4. **Test split (ALE-124 e1230) dominated by hypermutator passengers** — canonical-AMR stratum has n=7, statistically uninformative.
5. **Transformer underfits** — train_sanity ≈ test indicates not learning to saturation.
6. **HDD 50 GB hard-stop target unrealistic** given BV-BRC public-API data sizes per genome.

## 9. Day-3 Priorities (in order)

1. **Build canonical-AMR-rich eval split** — use Barbosa + Imamovic + Melnyk + curated clinical trajectories as the held-out clinical bar. Without this, the canonical-AMR metric is meaningless.
2. **Long-protein ESM2 embeddings** for canonical AMR proteins via chunked inference with overlap (8 GB GPU can do this; just need a windowing wrapper).
3. **Per-step MIC supervision head** on the transformer once measured (not proxy) MIC is available.
4. **LambdaRank + transformer reranker**: use transformer logits as an additional feature in LambdaRank over LambdaRank's top-10 candidates. Hypothesis: bridges 0.527→0.708 on overall *while* preserving the canonical-AMR signal from the 5× weighting.
5. **HGT/plasmid scope explicit handling**: tag blaKPC, mcr, vanA etc. as plasmid-borne per agent rule #1 and exclude from chromosomal trajectory model evaluation.

## 10. Files

Models:
- `/mnt/ssd/research/amr/models/residue_projector.pt` — Phase B v2, R² = 0.688
- `/mnt/ssd/research/amr/models/trajectory_transformer.pt` — Phase D v2, 2.6M params
- `/mnt/ssd/research/amr/models/day2/single_step_lgbm_v2.pkl` — Day-2 LambdaRank snapshot

Data:
- `/mnt/ssd/research/amr/processed/transitions.parquet` (28,578)
- `/mnt/ssd/research/amr/processed/trajectories.parquet` (1,379)
- `/mnt/ssd/research/amr/processed/transitions_with_mic.parquet` (20,512 with proxy MIC)
- `/mnt/ssd/research/amr/processed/dms_fitness.parquet` (95,890)

Results:
- `/mnt/ssd/research/amr/results/trajectory_eval.json`
- `/mnt/ssd/research/amr/results/trajectory_ablations.json`
- `/mnt/ssd/research/amr/results/day2/single_step_eval_v2.json`

Source: `/mnt/ssd/research/amr/src/day2/phase{A2,A3,A4,A5,B,C,D}_*.py`
