# Predicting the next resistance mutation

Given the ordered mutations a bacterial lineage has already accumulated under antibiotic selection, which mutation comes next? This repo holds the pipeline I built for that question: genome and trajectory data assembly, ESM2 protein representations, a LightGBM LambdaRank baseline, an autoregressive trajectory transformer, and a reranker over the two.

The clearest result is an ablation: removing the ordered mutation history from the transformer's input drops top-3 accuracy on the E. coli fluoroquinolone test split from 0.527 to 0.156. Ordered history, not the protein-language-model features, is what the model is actually using.

Project page: https://rohitravi.com/projects/amr-mutation-trajectory

## Data assembled

- 28,768 lab-evidence genomes from BV-BRC, phenotyped and annotated with AMRFinderPlus (a resistance-gene call on 98.81% of resistant-phenotype lab genomes; 18,193 annotation rows).
- 860 ordered directed-evolution trajectories / 28,028 mutation transitions, mostly from ALEdb projects (norfloxacin, tetracycline, ceftazidime, trimethoprim, chloramphenicol) plus Baym 2016 and Imamovic & Sommer 2013.
- ESM2-650M embeddings for 17,991 unique resistance protein sequences, 11,382 of them cached per residue (half precision, 8 GB RTX 4060 Ti; GyrA/ParC/RpoB handled by chunked inference in `src/day3/phase_b_chunked_esm2.py`).
- 95,722 single-mutant deep-mutational-scanning fitness measurements across TEM, VIM and DHFR families, z-scored within dataset, plus curated fluoroquinolone DMS from Pal 2015, Marcusson 2009 and Gullberg 2014.
- 1 of 28,028 transitions is plasmid-borne (`src/day3/phase_e_plasmid_scope.py`), so the models describe chromosomal point-mutation evolution specifically.

## Models

`src/gate6_baseline.py` — LightGBM LambdaRank over candidate mutations, 332 features (frequency priors, gene/drug-class context, ESM2 protein and residue PCA components, DMS fitness). On the E. coli fluoroquinolone split it gets top-3 = 0.705 over all transitions (`results/single_step_eval.json`).

`src/day2/phaseD_transformer_v2.py` — 4-layer causal decoder, d_model 128, 2.63 M parameters, vocabulary of 14,006 mutation tokens. Input per token: learned mutation embedding + projected ESM2 residue vector + DMS fitness scalar, conditioned on drug class and organism, cross-entropy with 5x weight on canonical-AMR mutations. The residue projector (`src/day2/phaseB_residue_projector_v2.py`) is an autoencoder that reconstructs 1280-d ESM2 residue vectors at R² = 0.688 from a 128-d latent, against 0.385 for PCA-32.

`src/day3/phase_d_unified.py` — three stages: LambdaRank scores the full vocabulary and keeps the top 10, the transformer rescores those 10 with autoregressive context, and a second LightGBM combines both scores with frequency priors and drug-class one-hots (23 features).

## Results

Ablations on the transformer, top-3 on the E. coli fluoroquinolone test split (`results/trajectory_ablations.json`, `results/trajectory_eval.json`):

| Variant | top-3 |
|---|---:|
| Full model | 0.527 |
| No canonical-AMR loss weighting | 0.548 |
| No ESM2 + DMS token features | 0.553 |
| No autoregressive context | 0.156 |

Dropping ESM2 and DMS features, or the canonical weighting, changes nothing or helps slightly. Dropping the history costs 0.37. That is the finding I would defend.

Day-3 canonical-AMR evaluation, n=571 held-out canonical-AMR transitions across 14 drug classes (`results/day3/phase_d_results.json`):

| Scoring | top-1 | top-3 | n |
|---|---:|---:|---:|
| LambdaRank, full vocabulary | 0.021 | 0.054 | 571 |
| Transformer, full vocabulary | 0.000 | 0.000 | 571 |
| Reranker, unconditional equivalent | — | ~0.051 | 571 |
| LambdaRank, on the top-10-contains-answer subset | 0.167 | 0.431 | 72 |
| Transformer, same subset | 0.153 | 0.403 | 72 |
| Reranker, same subset | 0.139 | 0.403 | 72 |

The reranker only ever sees the 72 of 571 transitions (12.6%) where stage 1 put the true mutation in its top 10, so its 0.403 is a subset number, not a headline. On that subset plain LambdaRank ranking scores 0.431 and the transformer alone 0.403: the reranker does not beat either component. Unconditionally it lands at about 0.051 against 0.054 for LambdaRank alone. Within the subset the per-class counts are tiny (carbapenem 0.667 at n=9, rifamycin 0.400 at n=10, macrolide 0.182 at n=11), and six classes have n=0 because stage 1 never surfaced their true mutation.

Two other caveats worth stating. MIC supervision is a linear ramp interpolated between ALEdb endpoint measurements, so the Pearson 0.92 between predicted step and MIC is close to tautological. Macrolide canonical top-1 is 0, which most likely means ribosomal-RNA loci are mis-tokenized, since that is where most macrolide resistance lives.

## Repo layout

```
src/gate0_setup.py .. gate6_baseline.py   data acquisition, annotation, trajectories,
                                          ESM2 embeddings, DMS assembly, LambdaRank baseline
src/day2/                                 residue projector, LambdaRank v2, trajectory transformer,
                                          trajectory expansion and fluoroquinolone DMS curation
src/day3/                                 AMRFinder extension, chunked ESM2, canonical-AMR split,
                                          reranker, plasmid scope tagging
results/                                  metrics JSON for every eval above
models/                                   transformer, residue projector, LambdaRank and reranker
```

Scripts read and write under `work/` (`work/raw`, `work/processed`, `work/embeddings`, `work/models`, `work/results`), which is gitignored; set that up as a symlink to wherever you keep the data. `models/` holds the trained checkpoints: `trajectory_transformer.pt` (11 MB), `residue_projector.pt` (15 MB), `single_step_lgbm.pkl`, and the Day-3 retrains under `models/day3/`. ESM2 embeddings are not bundled; regenerate with `src/gate4_esm2_embeddings.py` (about 17 minutes on an RTX 4060 Ti).

## Data sources

- BV-BRC genome metadata, phenotypes and protein FASTAs, pulled through the public API by `src/gate1_bvbrc.py` and `src/day2/phaseA5_assemblies.py`.
- AMRFinderPlus and its database, installed separately; point `AMRFINDER_BIN` at the binary.
- ALEdb (aledb.org) mutation tables and MIC endpoints, `src/day2/phaseA2_aledb_mic.py`.
- MaveDB and publication supplements for DMS data, `src/gate5_dms.py` and `src/day2/phaseA4_fq_dms.py`.
- ESM2-650M weights via `fair-esm`.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p work          # or: ln -s /path/to/data work

python src/gate1_bvbrc.py
python src/gate2_card.py              # AMRFinderPlus annotation
python src/gate3_trajectories.py
python src/gate4_esm2_embeddings.py   # GPU
python src/gate5_dms.py
python src/gate6_baseline.py          # LambdaRank baseline

python src/day2/phaseB_residue_projector_v2.py   # GPU, ~20 min
python src/day2/phaseD_transformer_v2.py         # GPU, ~3 h, writes the ablation table

python src/day3/phase_c_canonical_split.py
python src/day3/phase_d_unified.py
```

## Notes

The gap between the overall metric (top-3 0.705 for LambdaRank) and the canonical-AMR metric (0.054) is the honest story of this dataset: 86% of the trajectories come from one hypermutator fluoroquinolone experiment, so most transitions are passenger mutations that are easy to rank by frequency, and the clinically meaningful canonical-AMR transitions are rare and hard. There is no held-out-organism or held-out-drug-class evaluation here, so every number is on data chromosomally similar to the training set.

## License

MIT, see [LICENSE](LICENSE).
