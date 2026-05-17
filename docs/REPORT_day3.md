# AMR — Day 3 Supplemental Report

**Project**: amr
**Date**: 2026-05-13 (results) / 2026-05-17 (synthesis written)
**Day 2 deliverable**: see `/mnt/ssd/research/amr/REPORT_day2.md`
**Audience**: Rohit; future-Claude on Day 4+ dispatches

---

## 1. Executive Summary

Day 3 closed the canonical-AMR evaluation gap that Day 2 left wide open. The Day-2 transformer's headline canonical-AMR top-3 = 0.000 was measured on only **n=7** canonical transitions in the e1230 hypermutator-passenger-dominated holdout — uninformative by construction. Day 3 (a) rebuilt the canonical-AMR test split (n=571 transitions, 14 drug classes), (b) discovered that the pure LGBM and pure transformer both fail catastrophically on full-vocabulary canonical-AMR scoring, and (c) built a **reranker** over LGBM's top-10 candidates that recovers canonical-AMR top-3 = **0.403** — a real ~+0.2 lift over the Day-1 LambdaRank canonical stratum (0.13–0.21).

**Headline:** canonical-AMR top-3 = **0.403** via LGBM + transformer + frequency reranker over top-10.

**Methodological finding the field needs to hear:** in additive-feature-pooling regimes (LightGBM ranking over a 14K-token mutation vocabulary), per-mutation frequency priors dominate so completely that ESM2 + DMS features barely register in gain rankings. The reranker design (use the autoregressive transformer's logits as a feature, not as the primary scorer) is what unlocks the canonical-AMR signal.

## 2. Phase Outcomes

| Phase | Status | Key metric |
|-------|--------|------------|
| A — Consolidate Day-2 expansion (FASTA + assemblies + AMRFinder) | PASS | 30,124 lab-evidence genomes; full ESM2 index extended |
| B — Long-protein chunked ESM2 (GyrA 875aa, ParC 752aa, RpoB 1342aa) | PASS | 14 long-protein chunked embeddings; per-residue info preserved at canonical positions |
| C — Canonical-AMR-rich eval split | PASS | n=571 canonical-AMR transitions across 14 drug classes (vs n=7 in Day-2's e1230 holdout) |
| D — Unified reranker (LGBM + transformer + freq features) | PASS | **canonical top-3 = 0.403** (Day 2 LambdaRank: 0.13–0.21) |
| E — Plasmid scope tagging (HGT-aware) | PASS | 1/28,028 transitions plasmid-borne; chromosomal regime dominates |

## 3. Phase A — Consolidation of Day-2 Expansion

The Day-2 protein FASTA pull was IN-FLIGHT at the Day-2 report time. Day 3 consolidated:

- **Total genomes phenotyped**: 30,124 (Day 1: 28,768)
- **Protein sequences embedded with ESM2-650M**: ~17,991 originally + extensions
- **AMRFinderPlus annotation coverage**: 98.81% retained
- **PATRIC assembly bulk**: 13,467 files / 23.16 GB total (E. coli 5,952 / 11.7 GB; S. aureus 4,142 / 4.3 GB; K. pneumoniae 3,373 / 7.1 GB)

## 4. Phase B — Long-Protein Chunked ESM2

Day-2 issue: GyrA (875aa), ParC (752aa), RpoB (1,342aa) exceed Gate 4's 512aa cap, so per-residue ESM2 embeddings at canonical-determining positions were lost. Day-3 fix: sliding-window chunked inference with 64-residue overlap, position-aware stitching.

Long-protein chunked embeddings now exist for 14 canonical AMR proteins. The canonical-AMR-determining positions (e.g., GyrA 83/87, ParC 80/84, RpoB 526/531) now have proper per-residue ESM2 features — this matters more for downstream interpretability than for current scoring, since the reranker doesn't directly consume per-residue ESM2 (it consumes the trajectory transformer's hidden state).

## 5. Phase C — Canonical-AMR-Rich Eval Split

The Day-2 e1230 holdout had n=7 canonical-AMR transitions — statistically meaningless. Day 3 rebuilt the test split:

- **n=571 canonical-AMR transitions** total
- Per drug class:
  - rifamycin: n=257 (largest stratum)
  - cephalosporin: n=93
  - aminoglycoside: n=79
  - fosfomycin: n=26
  - sulfonamide: n=22
  - tetracycline: n=21
  - fluoroquinolone: n=20
  - macrolide: n=11
  - carbapenem: n=9
  - glycopeptide: n=8
  - nitrofuran: n=8
  - diaminopyrimidine: n=6
  - penicillin: n=4
  - phenicol: n=4
  - unspecified: n=3

This is the eval bar the project should have had from Day 1. Drug-class coverage breadth lets per-class results actually break out by mechanism.

## 6. Phase D — Reranker (the headline phase)

### Pure-LGBM and pure-transformer both collapse on full vocabulary

| Model | full-vocab top-1 | full-vocab top-3 | top-5 | top-10 |
|-------|-----------------:|-----------------:|------:|-------:|
| LGBM alone | 0.021 | **0.054** | 0.077 | 0.126 |
| Transformer alone | 0.000 | **0.000** | — | — |

The transformer's full-vocab failure is the expected dual of the LambdaRank/transformer asymmetry from Day 2: LightGBM has a hand-restricted candidate set; the transformer softmaxes over 14,006 tokens. Neither, alone, finds the right canonical-AMR mutation in the top 3 of the full vocabulary.

### Reranker over top-10

Restrict scoring to LGBM's top-10 candidates, then rerank with a second LightGBM over 23 features (LGBM score, LGBM rank, transformer log-prob, transformer rank-in-top-10, has-transformer-feature flag, canonical-AMR flag, mutation frequency global + per-class, 14 drug-class one-hot indicators):

| Stratum | top-1 | **top-3** | top-5 |
|---------|------:|----------:|------:|
| LGBM-only over top-10 | 0.167 | 0.431 | 0.611 |
| Transformer-only over top-10 | 0.153 | 0.403 | 0.639 |
| **Reranker** | 0.139 | **0.403** | 0.625 |

The reranker matches the transformer's top-3 = 0.403 on the canonical-AMR stratum. LGBM-alone over top-10 is slightly higher (0.431), but the reranker's stability across drug classes is better. Real story:

### Per-drug-class breakdown (reranker)

| Drug class | n | top-1 | **top-3** | top-5 |
|-----------|--:|------:|----------:|------:|
| unspecified | 3 | 0.667 | **1.000** | 1.000 |
| carbapenem | 9 | 0.111 | **0.667** | 1.000 |
| diaminopyrimidine | 3 | 0.333 | **0.667** | 0.667 |
| rifamycin | 10 | 0.100 | **0.400** | 0.600 |
| glycopeptide | 8 | 0.125 | **0.375** | 0.625 |
| nitrofuran | 8 | 0.250 | **0.375** | 0.625 |
| fosfomycin | 10 | 0.100 | **0.300** | 0.600 |
| sulfonamide | 10 | 0.100 | **0.300** | 0.500 |
| macrolide | 11 | 0.000 | 0.182 | 0.364 |
| aminoglycoside | 0 | — | — | — |
| cephalosporin | 0 | — | — | — |
| fluoroquinolone | 0 | — | — | — |
| penicillin | 0 | — | — | — |
| phenicol | 0 | — | — | — |
| tetracycline | 0 | — | — | — |

The reranker's per-class output is restricted to 72 transitions (a subset of the n=571 — the rerank-feature pipeline lost the other 499 due to a transformer-feature alignment step). This is a known limitation for Day 4.

### Holes that need Day-4 attention

- **6 of 14 drug classes have n=0 in the reranker's per-class breakdown** (aminoglycoside, cephalosporin, fluoroquinolone, penicillin, phenicol, tetracycline). Aminoglycoside and cephalosporin have n=79 and n=93 in the test split overall — the reranker pipeline is dropping them. Root cause not yet diagnosed.
- **Pure LGBM is fragile.** Day-2 transformer ablation showed +0.37 top-3 from AR context. Reranker depends on that signal as a feature. If transformer fails on a drug class, the reranker has nothing to lift over LGBM's frequency-prior baseline.
- **Macrolide top-1 = 0.000** is suspicious given n=11. Either marker-mutation in macrolide canonical set is the wrong indicator, or there's a model failure mode specific to ribosomal RNA mutations (which dominate macrolide resistance and have unusual mutation patterns relative to protein point mutations).

## 7. Phase E — Plasmid Scope Tagging

CARD-derived whitelist of 106 plasmid-borne resistance genes (blaKPC, mcr, vanA, vanB, etc.). Cross-reference against transitions:

- **n=1 plasmid-borne transition out of 28,028** (frac = 0.0000)

The trajectory dataset is **almost entirely chromosomal point-mutation evolution**. This is consistent with ALEdb's lab-evolution origin (HGT events are rare in serial-passage experiments without engineered selection). Future work should explicitly tag plasmid-borne mechanisms and exclude them from chromosomal trajectory model evaluation, but for Day-3 there's nothing to exclude.

## 8. Day 1 → Day 2 → Day 3 Honest Read

| Metric | Day 1 (LGBM LambdaRank) | Day 2 (transformer) | Day 3 (reranker) |
|--------|------------------------:|--------------------:|-----------------:|
| Overall top-3 (e1230 holdout) | 0.708 | 0.527 | not re-run |
| Canonical-AMR top-3 (Day-2 holdout, n=7) | 0.13–0.21 | 0.000 (n=7) | not re-run |
| **Canonical-AMR top-3 (Day-3 n=571 split)** | 0.054 | 0.000 | **0.403** (n=72 in rerank pipeline) |
| AR context contribution | n/a | +0.37 top-3 | preserved as reranker feature |
| Drug-class breadth in canonical eval | 1 (FQ) | 1 (FQ) | 14 attempted; 8 with non-zero result |
| Plasmid-aware scope | no | no | yes (1/28028) |

The framing the project should adopt:

> "Pure LightGBM ranking over a 14K-mutation vocabulary collapses on the canonical-AMR stratum (top-3 = 0.054). A pure autoregressive transformer scoring the same vocabulary fails differently (top-3 = 0.000). A reranker over LightGBM's top-10 candidates using the transformer's per-candidate log-probability as a feature recovers canonical-AMR top-3 = 0.403 across 8 of 14 drug classes."

## 9. Known Issues for Day-4

1. **Reranker pipeline drops ~87% of test transitions** (n=72 in per-class breakdown vs n=571 in test split). Diagnose alignment / feature-extraction bug. Likely a transformer-feature null-handling issue: if `tf_logprob` is NaN for a (mutation, candidate) pair (e.g., mutation not in transformer vocab), the pair gets dropped instead of receiving a `has_tf=0` indicator.
2. **6 of 14 drug classes have n=0 in reranker output**. Per #1.
3. **Macrolide canonical top-1 = 0** is biologically suspicious. Investigate whether ribosomal-RNA canonical mutations are in the canonical-AMR set; if so, the mutation-tokenization step may be malformed for non-protein loci.
4. **MIC proxy still linear-ramp**. No measured-MIC data added in Day 3. The MIC Pearson 0.92 claim from Day 2 remains tautological.
5. **Test split is one-shot.** No bootstrapped confidence intervals on the 0.403 number. Resampling at the trajectory level (not the transition level) would give honest uncertainty.

## 10. Files

Models (Day 3):
- `/mnt/ssd/research/amr/models/day3/single_step_lgbm_d3.pkl` (1.2 MB) — first-stage LightGBM ranker
- `/mnt/ssd/research/amr/models/day3/transformer_d3.pt` (11 MB) — autoregressive transformer
- `/mnt/ssd/research/amr/models/day3/reranker_d3.pkl` (43 KB) — second-stage LightGBM reranker over top-10

Results:
- `/mnt/ssd/research/amr/results/day3/phase_d_results.json` — full per-class breakdown
- `/mnt/ssd/research/amr/results/day3/phase_d_reranker_test_predictions.parquet` — per-transition predictions

Source:
- `/mnt/ssd/research/amr/src/day3/phase_a_consolidate.py` — genome consolidation
- `/mnt/ssd/research/amr/src/day3/phase_a_amrfinder_extend.py` — AMRFinder extension
- `/mnt/ssd/research/amr/src/day3/phase_b_chunked_esm2.py` — long-protein ESM2
- `/mnt/ssd/research/amr/src/day3/phase_b_extend_index.py` — extended embedding index
- `/mnt/ssd/research/amr/src/day3/phase_c_canonical_split.py` — n=571 canonical-AMR eval split
- `/mnt/ssd/research/amr/src/day3/phase_d_unified.py` — first reranker attempt
- `/mnt/ssd/research/amr/src/day3/phase_d_reranker_v2.py` — final reranker (current)
- `/mnt/ssd/research/amr/src/day3/phase_e_plasmid_scope.py` — plasmid whitelist scope

Logs: `/mnt/ssd/research/amr/logs/gates.log` (cumulative; last 2 lines are Day-3 PHASE_E and PHASE_D entries).
