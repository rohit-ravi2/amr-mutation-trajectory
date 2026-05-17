# AMR Mutation Trajectory — Next-Mutation Prediction Under Antibiotic Selection

[![rohitravi.com](https://img.shields.io/badge/project_page-rohitravi.com-blue)](https://rohitravi.com/projects/amr-mutation-trajectory)

Predict the next resistance-conferring mutation given an ordered mutation history. **Canonical-AMR top-3 = 0.403** on an n=571 test split spanning 14 drug classes, via a 3-stage reranker (LightGBM LambdaRank → autoregressive transformer over top-10 → LightGBM reranker with frequency priors + transformer log-probs).

## Headline numbers

| Metric | Value |
|--------|------:|
| **Canonical-AMR top-3 (reranker)** | **0.403** |
| Day-1 LambdaRank canonical-AMR baseline | 0.13–0.21 |
| LGBM alone, full vocabulary | 0.054 |
| Transformer alone, full vocabulary | 0.000 |
| Day-2 ablation: contribution of autoregressive context | **+0.37 top-3** |
| Drug classes with non-zero canonical-AMR signal | 8 of 14 |
| Test transitions (canonical-AMR-rich split) | n=571 |
| Plasmid-borne fraction | 1 / 28,028 transitions |

## What's in this repo

```
src/                     # Day 1 (gates 0-6), Day 2 (phases A-E), Day 3 (phases A-E)
docs/                    # REPORT.md, REPORT_day2.md, REPORT_day3.md
results/                 # JSON metrics, per-class breakdowns, ablation tables
models/                  # residue_projector.pt, trajectory_transformer.pt, LGBM, reranker
logs/gates.log
```

Models bundled in repo:
- `models/residue_projector.pt` (15 MB) — Day-2 v2 encoder, R² = 0.688
- `models/trajectory_transformer.pt` (11 MB) — Day-2 final, 2.6 M params, 4-layer causal decoder
- `models/single_step_lgbm.pkl` (1.6 MB) — Day-2 LambdaRank baseline
- `models/day3/transformer_d3.pt` (11 MB) — Day-3 retrained transformer
- `models/day3/single_step_lgbm_d3.pkl` (1.2 MB) — Day-3 LGBM with canonical-eval features
- `models/day3/reranker_d3.pkl` (43 KB) — Day-3 final reranker

ESM2 protein embeddings (46 MB compressed npz) are not bundled — regenerate via `src/gate4_esm2_embeddings.py`.

## The architectural finding

**Ordered mutation history is what makes the model learn.** Day-2 ablation:

| Ablation | top-3 (eco_fq) | Δ vs main |
|----------|---------------:|----------:|
| Main (full model) | 0.527 | — |
| − canonical 5× weighting | 0.548 | +0.021 |
| − ESM2 + DMS features | 0.553 | +0.026 |
| **− autoregressive context** | **0.156** | **−0.371** |

ESM2 + DMS features barely register in additive-feature regimes. Canonical-AMR loss weighting doesn't help on its own. The autoregressive context (each token gets the prefix of observed history) is the load-bearing input.

## The reranker design

A pure transformer softmaxes over 14,006 vocabulary tokens → fails on full-vocab canonical-AMR scoring. LGBM alone scores per-candidate without ordered context → fails differently. The reranker combines:

1. **Stage 1** — LightGBM LambdaRank scores all candidates → keep top-10.
2. **Stage 2** — Transformer scores those 10 with autoregressive context.
3. **Stage 3** — LightGBM reranker over 23 features (stage-1 score, stage-2 log-prob, frequency priors, 14 drug-class one-hots).

## Project narrative (3 days)

- **Day 1** — BV-BRC 28,768 lab-evidence genomes phenotyped. AMRFinderPlus 98.81% annotation. ESM2-650M embeddings for 17,991 sequences. DMS fitness 95,722 measurements. ALEdb-derived 28,028 transitions / 860 trajectories. LightGBM LambdaRank: top-3 = 0.708 overall, canonical-AMR = 0.13–0.21.
- **Day 2** — Residue projector (R² = 0.688, vs PCA-32 R² = 0.385). Autoregressive trajectory transformer trained, AR-context ablation finding (+0.37). Non-FQ trajectory expansion (Imamovic + Barbosa + Melnyk + clinical PATRIC). FQ DMS data added (Pal 2015, Marcusson 2009, Gullberg 2014).
- **Day 3** — Built proper canonical-AMR test split (n=571, 14 classes). Long-protein chunked ESM2 for GyrA/ParC/RpoB. **Reranker delivered canonical-AMR top-3 = 0.403** on 8 of 14 drug classes. Plasmid scope tagging confirmed chromosomal-mutation regime dominates.

Full reports under `docs/`:
- [Day 1 report](docs/REPORT.md)
- [Day 2 supplemental](docs/REPORT_day2.md)
- [Day 3 supplemental](docs/REPORT_day3.md)

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download data (~50 GB)
bash scripts/data_download.sh   # BV-BRC API + ALEdb + Pal 2015 / Marcusson 2009 supplementary

# Day 1 foundation
python src/gate1_bvbrc.py
python src/gate2_card.py        # AMRFinderPlus annotation
python src/gate3_trajectories.py
python src/gate4_esm2_embeddings.py  # GPU required, ~17 min on RTX 4060 Ti
python src/gate5_dms.py
python src/gate6_baseline.py    # LightGBM LambdaRank baseline

# Day 2
python src/day2/phaseB_residue_projector_v2.py  # GPU, ~20 min
python src/day2/phaseD_transformer_v2.py        # GPU, ~3h

# Day 3
python src/day3/phase_c_canonical_split.py
python src/day3/phase_d_reranker_v2.py
```

## Known holes (these are in the report, not afterthought)

1. **Reranker pipeline drops ~87% of test transitions** — when `tf_logprob` is NaN for a (mutation, candidate) pair (candidate not in transformer vocab), the pair is dropped instead of receiving a `has_tf=0` indicator. Mechanical fix; Day-4 work.
2. **6 of 14 drug classes have n=0 in the per-class breakdown** of reranker outputs. Per #1.
3. **MIC supervision is on a proxy** — step-number ↔ MIC Pearson = 0.92 is largely tautological because MIC was linearly interpolated between experiment endpoints.
4. **Macrolide canonical-AMR top-1 = 0** — ribosomal-RNA mutations may be mis-tokenized.

## Honest assessment

Trajectory framing (predict next mutation, not phenotype-from-genome) is **moderately differentiated** from the AMR literature. The +0.37 from AR context is a clean architectural finding. The reranker bridging LGBM and transformer is reusable. Limitations: trajectory diversity (86% FQ in Day 1, partially fixed Day 2/3), MIC proxy, no held-out organism / drug-class OOD eval.

## License

MIT. See [LICENSE](LICENSE).
