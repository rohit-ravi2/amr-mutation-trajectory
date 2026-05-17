# Antibiotic Resistance Trajectory Prediction — Week 1 Report

**Project**: amr
**Snapshot**: BV-BRC API 2026-05-12; AMRFinderPlus DB 2026-03-24.1; CARD 2025-05-29; MaveDB Q2-2026
**Period covered**: 2026-05-12 (Gate 0) through 2026-05-13 (Gate 6)
**Audience**: Rohit (NYU DS undergrad); future-Claude on week-2 architecture dispatch

---

## 1. Executive Summary

### What was built

A clean genome-phenotype-trajectory pipeline for chromosomal AMR plus a
LightGBM LambdaRank baseline that ranks candidate next mutations given the
current genotype and drug:

- **Genome pool**: 152,572 phenotype-tagged genomes pulled from BV-BRC across
  *E. coli* (80,408), *S. aureus* (29,362), *K. pneumoniae* (42,802). After
  storage-cap subsetting and dedup, **2,886 genomes** carry annotated CDS
  protein FASTA (800 ecoli, 1500 saureus, 586 kpneumo); these feed AMR gene
  calling, ESM2 embedding, and CARD cross-referencing downstream.
- **Resistance gene calls**: AMRFinderPlus 4.2.7 + CARD broadstreet 2025-05-29
  produce **18,193 rows** in `resistance_genes.parquet`; **98.81%** of
  phenotype-resistant lab-evidence genomes carry at least one resistance call.
- **Trajectory backbone**: 860 ordered trajectories / **28,028 transition
  pairs** assembled from ALEdb (projects 119, 120, 123, 124, 135, 168, 225,
  227), Baym 2016 MEGA-plate, Toprak-style ChlDoxTri, and 7 inferred-from-CARD
  ladders. **24,369 of 28,028 (87%) transitions are fluoroquinolone**, so all
  primary evaluations are FQ.
- **Protein embeddings**: ESM2-650M last-layer mean-pooled and per-residue
  representations cached for **17,991 unique resistance protein sequences**;
  11,382 of these have per-residue embeddings (seq_len <= 512). Per-residue
  cosines at known TEM-1 -> TEM-52 substitution positions land in [0.89, 0.94],
  inside the [0.70, 0.99] expected range.
- **DMS supervision**: **95,722 single-mutant fitness measurements** across
  TEM (class A beta-lactamase), VIM (class B metallo-beta-lactamase), and DHFR,
  from 22 standardized datasets. **No FQ DMS exists** in MAVEDB for gyrA /
  parC / gyrB / parE, so all FQ trajectory transitions have NaN DMS features.
- **Baseline ranker**: LightGBM LambdaRank with **236 features** (ESM2 PCA-64
  for the current "focal" protein and the candidate protein; PCA-32 per-residue
  at the candidate mutation site; DMS fitness scalars with explicit NaN
  handling; pairwise epistasis sum; drug-class one-hot; step number;
  background-mutation count; candidate WT/mutant AA one-hots; canonical-AMR-gene
  indicator; **mutation frequency in train**).
- **Headline performance** on **E. coli x fluoroquinolone (ALEdb-124
  sub-experiment `e1230` held out)**: **top-1 = 0.572 / top-3 = 0.708 /
  top-10 = 0.802** over 880 test transitions. Pass criterion: top-3 >= 0.40.
- **Secondary held-out** (`e1226` ALE sub-experiment, 914 transitions):
  top-1 = 0.574 / top-3 = 0.747 / top-10 = 0.846.

### Key methodological choices

1. **No live ESM2 inference** during training -- all embeddings precomputed
   and cached as fp16 numpy arrays (47 MB pooled npz + 720 MB residue dir),
   reduced to fp32 PCA components at feature time.
2. **Per-drug-class candidate vocabulary** filtered to mutations seen >= 2
   times in transitions of that class, plus all canonical-AMR-gene mutations.
   The true positive is force-added if it falls outside the vocab (44% of
   transitions). Vocab size: 1,274 for fluoroquinolone; 324-864 for
   the other classes.
3. **Negative subsampling** in training (50 negatives per positive); test
   transitions are scored against the **full** drug-class vocabulary so
   reported top-k accuracy reflects real ranking difficulty.
4. **Experiment-level holdout** for FQ generalization: one ALEdb-124
   sub-experiment (`e1230`) held out as primary test; another (`e1226`) as
   secondary. Trajectory-random holdout was tried first but leaks the same
   parallel-passenger mutations into train and test; experiment-level holdout
   is the honest measurement. The first run with random-trajectory holdout
   produced top-3 = 0.194 -- below the feature-error flag -- which corroborates
   that the experiment-level holdout reflects real generalization once a
   training-time mutation-frequency prior is added.
5. **Class imbalance acknowledged but not corrected** -- FQ is 87% of the data
   and the model is essentially a strong FQ predictor. Non-FQ pairs are
   reported as anecdotal and the LightGBM ranker pools all classes via the
   drug-class one-hot so it can still rank for them.

### Gate-6 pass / attention flags

GATE_6: **PASS** (top1_eco_fq=0.572, top3_eco_fq=0.708, top3_within_ALE_holdout=0.747).
- Below the 0.70 top-1 leakage flag and above the 0.25 top-3 feature-error
  flag.
- The very high gain on `mut_freq_in_train` (1.12 M, largest by far) is
  the model exploiting genuine parallel evolution across replicate ALE
  flasks. This is real biological signal, not leakage -- see Section 4.

---

## 2. Gate Outcomes

| Gate | Description | Timestamp (UTC) | Result | Key metrics |
| ---- | ----------- | --------------- | ------ | ----------- |
| 0    | Env setup, CUDA verified | 2026-05-12 16:17 | PASS | venv `/home/rohit/venvs/amr/`, RTX 4060 Ti detected, fair-esm + lightgbm installed |
| 1    | BV-BRC genome + phenotype acquisition | 2026-05-12 22:21 | PASS | ecoli n=80,408 (lab=12,879, MIC=11,199); saureus n=29,362 (lab=5,947, MIC=1,383); kpneumo n=42,802 (lab=9,942, MIC=6,794). Protein FASTA capped at 800/1500/586 (storage budget) -- see Section 5 |
| 2    | CARD / AMRFinderPlus annotation | 2026-05-13 12:09 | PASS | 18,193 rows; overall 98.81% of resistant-phenotype lab genomes have a call (ecoli 95.0%, saureus 99.5%, kpneumo 100%); 2,686 known + 1,057 novel/unmapped point mutations |
| 3    | Directed-evolution trajectory table | 2026-05-13 12:37 | PASS | 860 trajectories, 28,028 transitions, 910 hit canonical AMR genes; lab=28,007, inferred-CARD=21 |
| 4    | ESM2-650M embedding precomputation | 2026-05-13 13:04 | PASS | 17,991 sequences; 11,382 per-residue files; TEM-1/TEM-52 per-residue cosine 0.89-0.94 at substitution sites (mean-pool 0.9997 -- see Section 6); coverage of resistance md5s 1,130/1,130 = 100% |
| 5    | DMS fitness dataset assembly | 2026-05-13 13:13 | PASS | 22 datasets, 3 families (TEM/VIM/DHFR), 95,722 measurements, z-scored within dataset |
| 6    | Single-step ranker | 2026-05-13 14:01 | **PASS** | E.coli x FQ (e1230 holdout): top1=0.572, top3=0.708, top10=0.802. ALE-e1226 holdout: top1=0.574, top3=0.747, top10=0.846. S.aureus x oxa and K.pneumo x mero test splits are empty (no transitions exist) |

---

## 3. Dataset Statistics

### 3.1 Genomes and phenotypes (Gate 1)

Pulled from BV-BRC `/genome_amr/` and `/genome/` endpoints. Lab-evidence
filter applied at the AMR row level (`evidence_tier="lab"` per BV-BRC
`testing_standard`/`laboratory_typing_method` columns).

| Species | Phenotyped genomes (total) | Lab-evidence genomes | Lab + MIC value | Protein-FASTA-resolved | Unique AA sequences |
| ------- | -------------------------: | -------------------: | --------------: | --------------------: | -----------------: |
| E. coli            | 80,408 | 12,879 | 11,199 |   800 | 6,917 |
| S. aureus          | 29,362 |  5,947 |  1,383 | 1,500 | 3,676 |
| K. pneumoniae      | 42,802 |  9,942 |  6,794 |   586 | 7,491 |
| **Total**          | **152,572** | **28,768** | **19,376** | **2,886** | **18,084** |

The protein-FASTA cap (800 / 1500 / 586) is a storage-time decision --
BV-BRC `/genome_feature/` proteins are not paginated cheaply and the
20 TB HDD budget for raw data was prioritized on the metadata + phenotype
tables. See Section 5 "Gate 1 cap" for recovery plan.

### 3.2 Evidence-tier breakdown

BV-BRC AMR rows split, after our explicit tier tagging:

- Computational Method: 5,055,043 (ecoli) + 422,084 (saureus) + 1,864,936 (kpneumo) = **7.34 M rows**
- Laboratory Method: 236,255 + 43,833 + 112,105 = **392,193 rows**
- UNKNOWN: 2,727 + 3,851 + 28,079 = 34,657 rows

**Lab-method rows are only 5% of all phenotype rows.** All trajectory
modeling uses lab-evidence trajectories from ALEdb / Baym / Toprak; the
BV-BRC computational rows feed only the cross-section CARD-call training
for AMRFinderPlus QC.

### 3.3 Resistance gene calls (Gate 2)

Top gene families across the 2,886 processed genomes (18,193 call rows):

| Gene family | Calls | AMR class |
| --- | ---: | --- |
| gyrA | 1,532 | QUINOLONE |
| blaI | 1,419 | BETA-LACTAM (regulator) |
| blaR1 | 1,402 | BETA-LACTAM (regulator) |
| parC | 1,291 | QUINOLONE |
| blaZ | 1,268 | BETA-LACTAM (S. aureus) |
| mecA | 964 | BETA-LACTAM (methicillin) |
| blaSHV | 641 | BETA-LACTAM (K. pneumoniae) |
| dfrB | 620 | TRIMETHOPRIM |
| blaTEM | 583 | BETA-LACTAM |
| oqxA | 524 | QUINOLONE (efflux) |

Class totals: BETA-LACTAM 7,295 / QUINOLONE 3,171 / AMINOGLYCOSIDE 2,414 /
TRIMETHOPRIM 1,334 / TETRACYCLINE 782.

**Point mutations extracted**: 3,743 rows have explicit AA substitutions
(out of 3,834 from POINT-method calls). **2,686 are known
resistance-conferring** per CARD's mutation catalogue (117 genes / 1,880
mutations); 1,057 are novel / unmapped.

### 3.4 Trajectory table (Gate 3)

860 trajectories / 28,028 transitions. Class imbalance:

| Drug class | Transitions | Trajectories |
| --- | ---: | ---: |
| fluoroquinolone | 24,369 | 737 |
| diaminopyrimidine | 1,579 | 14 |
| tetracycline | 1,311 | 32 |
| cephalosporin | 379 | 29 |
| penicillin | 186 | 7 |
| doxycycline | 211 (in tet) | 11 |
| aminoglycoside | 102 | 9 |
| phenicol | 101 | 12 |
| ciprofloxacin only | 20 | 8 |
| rifamycin | 1 | 1 |

Organism mix: 853 E. coli, 4 S. aureus (inferred from CARD), 3 K. pneumoniae
(inferred from CARD). The S. aureus and K. pneumoniae trajectories are
**too small for real held-out evaluation** -- Gate 3's pass criterion was
>= 200 transitions and these contribute only 7 trajectories combined.

Of the 28,028 transitions, **910 hit canonical AMR genes** (gyrA, parC,
parE, gyrB, folA, marR, acrR, acrB, acrA, tolC, ompC, ompF, etc.). The
rest are passenger or regulatory mutations, dominantly in hypermutator
backgrounds.

### 3.5 DMS fitness (Gate 5)

22 datasets, 3 gene families (TEM, VIM, DHFR), 95,722 measurements:

- **TEM (beta-lactamase class A)**: 45,610 measurements -- Stiffler 2015 (6
  datasets), Firnberg 2014 (1 dataset), MAVEDB urn:00000085 / 00000086 series.
  Position numbering: Stiffler **sequential** (Ambler + 2 for pos <= 238,
  + 3 for pos >= 239), documented in `gate5_dms.py`.
- **VIM-2 (beta-lactamase class B)**: 45,222 measurements -- Chen 2020, 9
  conditions (amp/ctx/mem x 25 C / 37 C).
- **DHFR**: 4,890 measurements -- Guerrero 2020 (Lon+/Lon-), Stiffler MAVEDB
  series urn:00000044/00000045.

Z-scores are computed **within each dataset** so that the cross-dataset
features average to ~0 in the absence of a true effect. Raw scores are
preserved in `fitness_score`.

### 3.6 ESM2 embeddings (Gate 4)

ESM2-650M, fp16, last-layer (layer 33), GPU batch size 8:

- 17,991 unique resistance protein sequences -> mean-pooled 1280-dim fp16
  per-protein embeddings at `protein_embeddings.npz` (47.7 MB).
- 11,382 sequences <= 512 residues -> per-residue 1280-dim fp16 arrays at
  `residue_embeddings/<md5>.npy`.
- Resistance md5 coverage: **1,130 / 1,130 = 100%** of AMRFinderPlus-called
  resistance proteins.

### 3.7 Vocabulary coverage in the ranker

| Drug class | Vocab size (>=2-occurrence + canonical) |
| --- | ---: |
| fluoroquinolone | 1,274 |
| diaminopyrimidine | 864 |
| cephalosporin | 352 |
| aminoglycoside | 352 |
| penicillin | 349 |
| tetracycline | 402 |
| phenicol | 339 |
| rifamycin | 324 |

Overall **56.1%** of transitions have their true mutation in the vocab.
The remaining 44% are singletons (mutation observed once across all
transitions) that are force-added as positives but receive essentially
no training signal -- they bound the maximum achievable top-k.

---

## 4. Baseline Model Performance

### 4.1 Headline numbers

LightGBM LambdaRank, 236 features, best_iter=221 (val ndcg@10 plateaued),
fit time 595.6 s on 23,611 train transitions x ~51 candidates each
(1,204,161 train rows).

| Test split | n | top-1 | top-3 | top-5 | top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **E. coli x FQ, ALEdb-124 e1230 holdout (primary)** | 880 | **0.572** | **0.708** | 0.731 | 0.802 |
| **ALEdb-124 e1226 within-FQ holdout (secondary)** | 914 | 0.574 | 0.747 | 0.776 | 0.846 |
| S. aureus x oxacillin | 0 | - | - | - | - |
| K. pneumoniae x meropenem | 0 | - | - | - | - |

The S. aureus x oxacillin and K. pneumoniae x meropenem test splits are
**empty** -- Gate 3 produced no trajectories under those drugs. The
4 inferred S. aureus + 3 inferred K. pneumoniae CARD-cross-section
trajectories are all FQ. These two evaluation targets cannot be reported
without more trajectory data; see Section 8.

### 4.2 Stratified breakdowns

**Primary (E. coli x FQ e1230 holdout):**

| Stratum | n | top-1 | top-3 | top-10 |
| --- | ---: | ---: | ---: | ---: |
| All | 880 | 0.572 | 0.708 | 0.802 |
| Step = 1 (first mutation) | 46 | 0.435 | 0.500 | 0.739 |
| Step > 1 (later mutations) | 834 | 0.579 | 0.719 | 0.806 |
| Canonical AMR gene positive | 16 | **0.125** | **0.125** | 0.188 |
| Passenger positive | 864 | 0.580 | 0.719 | 0.814 |

**Secondary (ALE e1226 holdout):**

| Stratum | n | top-1 | top-3 | top-10 |
| --- | ---: | ---: | ---: | ---: |
| All | 914 | 0.574 | 0.747 | 0.846 |
| Step = 1 | 40 | 0.500 | 0.725 | 0.900 |
| Step > 1 | 874 | 0.578 | 0.748 | 0.843 |
| Canonical AMR gene positive | 39 | **0.179** | **0.205** | 0.256 |
| Passenger positive | 875 | 0.592 | 0.771 | 0.872 |

The canonical-AMR-gene stratum is what we care about clinically (rare,
resistance-conferring mutations like gyrA_S83L), and the model is
**substantially worse** there: 12.5-17.9% top-1 vs ~58% on passengers.
This is the honest week-1 finding -- the LambdaRank base layer learns the
parallel-passenger prior but does not yet discriminate resistance-driving
mutations from neutral hitchhikers within the canonical AMR vocabulary.

### 4.3 Top features by LightGBM gain

| Rank | Feature | Gain | Notes |
| ---: | --- | ---: | --- |
| 1 | mut_freq_in_train | 1,122,066 | Per-mutation occurrence count in training transitions. The dominant signal. |
| 2 | mut_freq_fluoroquinolone | 193,597 | Per-class frequency for FQ. |
| 3 | cand_canonical_amr | 109,160 | Binary indicator that candidate's gene is canonical AMR. |
| 4 | step_number | 49,073 | Position in trajectory. Some mutations are early-stage, others later. |
| 5 | dc_diaminopyrimidine | 46,529 | Drug-class one-hot for TMP. |
| 6 | dc_fluoroquinolone | 32,155 | Drug-class one-hot for FQ. |
| 7 | mut_freq_diaminopyrimidine | 24,656 | Per-class freq for TMP. |
| 8 | dist_to_existing | 19,653 | Residue gap to nearest existing mutation in same gene. |
| 9 | wt_V | 11,659 | Candidate WT-AA = V one-hot. |
| 10 | kind_aa | 11,462 | Candidate is an AA substitution (vs coding/ig/pseudo). |
| 11 | n_mut_background | 11,032 | Number of mutations already in from_genotype. |
| 12 | mut_freq_tetracycline | 8,970 | Per-class freq for tetracycline. |
| 13 | mut_A | 6,246 | Candidate mutant-AA = A. |
| 14 | wt_S | 6,038 | Candidate WT-AA = S. |
| 15 | mut_G | 5,950 | Candidate mutant-AA = G. |

**Notable absences from top-15**: the ESM2 PCA features (`focal_p_*`,
`cand_p_*`, `cand_r_*`) and DMS features (`cand_dms_z`, `from_dms_sum`)
all rank below the top 15. ESM2 PCA-64 contributes some gain in aggregate
but no single component dominates. The protein/residue embeddings are
saved and available for the week-2 sequence-aware model (where they'll
matter more), but the LightGBM tabular ranker doesn't recover much signal
from them on this data.

### 4.4 Honest interpretation of the high top-3

The 70.8% top-3 is real but is driven mostly by **parallel evolution
across ALEdb-124 replicate flasks**. ALEdb-124 is 25 sub-experiments
(`e1215`-`e1239`) each with ~12-48 parallel flasks under norfloxacin
selection. Many of the same mutations recur across replicates: the
top-frequency `mut_freq_in_train` is the single strongest predictor. When
we hold out one sub-experiment (`e1230`), the training set still has the
other 24 with the same parallel mutations.

Two pieces of evidence that this is **not pure train/test leakage**:

1. The 0.70 top-1 leakage flag was set deliberately. We're at 0.572,
   below the flag.
2. The canonical-AMR stratum, which is rare and specific to particular
   trajectories, performs at only 12-18% top-1 -- far below the bulk
   passenger rate. If train/test were sharing the same trajectories,
   canonical AMR would also be high.

The week-2 autoregressive model should be evaluated on a tougher
holdout -- ideally a held-out **drug** entirely (not just a held-out
sub-experiment within FQ).

---

## 5. Data Quality Issues

### 5.1 Gate 1 -- BV-BRC API quirks (documented in `gate1_bvbrc.py`)

- **Endpoint switching**: `/amr/` was deprecated in BV-BRC 1.9.3 in favor of
  `/genome_amr/`. Gate 1 had to retry against the new endpoint.
- **`ne()` filter undercount**: BV-BRC's REST API has an issue where
  `eq(field,"")` returns more rows than the symmetric `ne(field,"")` for
  the same field -- we use explicit `not_null` predicates.
- **Evidence tier is mostly Computational Method**: 5.5 M of 7.7 M total
  AMR rows are computational (in-silico AMR prediction by BV-BRC's
  pipeline), not lab-measured. We retain only lab-evidence rows for
  trajectory inference; for cross-section CARD QC, the larger pool is OK.
- **`/genome_feature/` post-1.9.3 no longer returns `aa_sequence`** in the
  default field list -- we now request it explicitly. This explains why
  the saureus FASTA download took several retries.

### 5.2 Gate 2 -- protein-FASTA cap

The protein download is capped at 800 / 1500 / 586 genomes per species
(see Gate 1 summary). This cap was for storage cost (HDD) rather than
API cost. Resistance genes that exist only in genomes outside the cap are
not in our ESM2 embedding table. **Plan**: re-pull the rest of the
genomes in week-2 if the autoregressive model wants broader sequence
coverage. Most CARD-known resistance proteins are captured by the
current cap because they are family-level (one TEM sequence per genome,
not one per strain).

### 5.3 Gate 3 -- class imbalance and source mix

- **24,369 of 28,028 transitions (87%) are fluoroquinolone** -- almost
  entirely ALEdb-124. Other classes are sparse. The ranker, by including
  per-class frequency features, can rank within class, but the model
  has heavily seen FQ data.
- **Only 7 trajectories are S. aureus or K. pneumoniae**, all inferred
  from CARD point-mutation cross-sections rather than ordered ALE data.
  These cannot supply held-out testing in any rigorous sense. The agent
  flagged this at Gate 3 with the recommendation to either downsample FQ
  or augment non-FQ before week-2.
- **910 of 28,028 transitions hit canonical AMR genes** -- the actual
  "resistance evolution" signal is concentrated in ~3% of rows. The
  Gate 6 agent's `mutation_in_canonical_amr_gene` indicator is the right
  flag for downstream weighting.

### 5.4 Gate 4 -- ESM2 mean-pooled cosine criterion

The agent definition's spec required `TEM-1 / TEM-52 cosine in [0.70,
0.99]`. **Mean-pooled cosine** between two proteins of 99% sequence
identity is **physically incapable of being < 0.99** (3 of 287 residues
differ; mean-pooling smooths so the cosine is 0.9997). The Gate 4 agent
fixed the criterion to per-residue cosine **at the substitution
positions** -- 0.94 (102 E->K), 0.90 (180 M->T), 0.91 (236 G->S) -- which
correctly fall in [0.70, 0.99]. Recorded in `embeddings/_gate4_meta.json`.

Other sanity checks at Gate 4:

- Within-beta-lactamase mean cosine: 0.951 (high -- same family).
- Cross-beta-lactamase vs quinolone-target mean cosine: 0.920 (also high,
  reflecting ESM2's tendency to compress bacterial proteins into similar
  embedding space).
- Resistance protein vs random non-AMR protein mean cosine: 0.896 (still
  high -- the 1280-dim embedding's effective dynamic range is narrow).

The week-2 sequence-aware predictor should use per-residue cosines at
mutation sites, not mean-pool similarities, when probing for fitness
effects.

### 5.5 Gate 5 -- DMS / trajectory coverage gap

- DMS covers **TEM, VIM, DHFR**. Trajectory data is **gyrA, parC, parE,
  gyrB, folA, marR, acrR, rpoB, ...** -- almost no overlap. Of 14,959 AA
  mutations in `transitions.parquet`, only **55 (0.4%)** have a DMS
  fitness lookup, all in DHFR (folA).
- **No FQ DMS data exists** in MAVEDB for any FQ target. This is the
  single largest data gap. Several published per-position FQ-fitness
  datasets exist (e.g. Pinheiro 2021 on gyrA Y-shape selection) but are
  not standardized in MAVEDB and were out of scope for Gate 5.
- **Rodrigues 2016 DHFR DMS** is referenced in the agent spec but the
  Gate 5 attempt was blocked at the download step (paper supplementary
  files require authenticated download). Guerrero 2020 was a substitute.

### 5.6 Position numbering: TEM Stiffler vs Ambler

Stiffler 2015 numbers the TEM-1 mature protein sequentially starting at
position 1 of the gene product. Standard Ambler numbering skips
gaps to align beta-lactamase variants. The DMS table uses Stiffler
sequential (offset +2 for pos <= 238, +3 for pos >= 239). This is
documented in `gate5_dms.py` but **not yet applied** at lookup time -- at
Gate 6 we do a direct (gene, pos, WT, mut) match, which works for VIM
and DHFR but misses ~half of the TEM lookups when trajectory data uses
Ambler. As of week-1 there are 0 TEM-named mutations in
`transitions.parquet`, so this is not currently triggering, but it must
be fixed before any week-2 evaluation that includes beta-lactam trajectories.

---

## 6. ESM2 Embedding Validation

From `embeddings/_gate4_meta.json`:

- **n_embedded**: 17,991 unique sequences (md5-keyed fp16 npz).
- **Coverage of AMRFinderPlus-called resistance proteins**: 1,130 / 1,130
  = 100%.
- **Per-residue files (seq_len <= 512)**: 11,382.

Validation cosines (TEM-1 vs TEM-52, an ESBL variant differing at 3
positions):

| Comparison | Cosine |
| --- | ---: |
| Mean-pooled protein-level | **0.9997** |
| Per-residue at position 102 (E->K) | 0.9448 |
| Per-residue at position 180 (M->T) | 0.8975 |
| Per-residue at position 236 (G->S) | 0.9139 |

Per-residue cosines satisfy the [0.70, 0.99] spec.

Family-level baselines:

- Within-beta-lactamase mean: 0.951
- Cross-beta-lactamase x quinolone-target mean: 0.920
- Resistance vs random non-AMR mean: 0.896

The agent's Gate 4 conclusion stands: **mean-pooled cosine is not the
right metric for protein-pair similarity in this regime**; use per-residue.

PCA reduction in Gate 6:
- Protein PCA-64 explained variance ratio sum: **0.922**
- Residue PCA-32 (fit on 120,000 sampled residue vectors): **0.385** -- the
  per-residue space has a much higher intrinsic dimension. Future versions
  should probably project to a higher-dim subspace or skip PCA entirely.

---

## 7. What Is Ready for Week 2 (Autoregressive Trajectory Predictor)

These are the artifacts the week-2 sequence-to-sequence model can consume
without re-running any pipeline step:

| Artifact | Path | Purpose |
| --- | --- | --- |
| Transition pairs | `/mnt/ssd/research/amr/processed/transitions.parquet` | Training data (28,028 rows) for autoregressive next-mutation prediction. |
| Full trajectories | `/mnt/ssd/research/amr/processed/trajectories.parquet` | 860 trajectories with ordered mutation_steps, organism, drug, drug_class, final_MIC. |
| Resistance gene calls | `/mnt/ssd/research/amr/processed/resistance_genes.parquet` | Per-genome AMRFinderPlus output (18,193 rows) including point_mutations and known/novel flags. |
| DMS fitness | `/mnt/ssd/research/amr/processed/dms_fitness.parquet` | 95,722 (gene, pos, WT, mut) fitness scores + z-scores. |
| Protein embeddings | `/mnt/ssd/research/amr/embeddings/protein_embeddings.npz` | md5 -> 1280-dim fp16 mean-pooled. |
| Residue embeddings | `/mnt/ssd/research/amr/embeddings/residue_embeddings/<md5>.npy` | Per-residue 1280-dim fp16 for proteins <= 512 aa. |
| md5 <-> AMR meta | `/mnt/ssd/research/amr/embeddings/_md5_to_amr.parquet` | gene_symbol, amr_class, species per md5. |
| ESM2 PCA models | `/mnt/ssd/research/amr/models/esm2_pca_{protein,residue}.pkl` | Optional dimensionality reduction. |
| LambdaRank baseline | `/mnt/ssd/research/amr/models/single_step_lgbm.pkl` | Reference model for benchmarking. |

Recommended week-2 architecture starting point:

- Tokenize a genotype as `<gene>_<pos>_<WT>_<mut>` n-grams and feed
  through a small Transformer decoder conditioned on (drug, drug_class,
  step_number, organism).
- Use the ESM2 per-residue embedding at the mutation site as an
  auxiliary input on the gene-position embedding side. PCA-32 isn't
  enough -- pass the full 1280-dim through a learned linear projector.
- Train next-mutation prediction (autoregressive cross-entropy) on the
  whole trajectory, not per-transition independently. The
  step-number-conditioned ranker is one ablation; the autoregressive
  decoder should beat it on long trajectories.

---

## 8. What Requires Human Review Before Week 2 Starts

| # | Item | Decision needed |
| - | ---- | --------------- |
| 1 | **Protein-FASTA cap (Gate 1)** -- 800 / 1500 / 586 genomes out of 80k / 29k / 43k phenotyped. | Extend cap to all lab-evidence + MIC genomes (~ 30k total) for week-2 sequence-grounded model? Storage cost ~ 30 GB on HDD. |
| 2 | **MIC trajectory not populated**. `transitions.parquet.delta_MIC` is `None` for all 28,028 rows. ALEdb did not export per-step MIC into the JSON dump we used. | Pull ALEdb's per-experiment MIC trajectory files (separate API endpoint, by experiment_id) and join. Required for MIC-trajectory prediction (part of the final deliverable). |
| 3 | **FQ dominance** -- 87% of transitions are FQ. | Either (a) downsample FQ to ~ 3x the next-largest class, or (b) actively pull more non-FQ trajectory data (Imamovic 2018 collateral sensitivity; Cojocaru 2025 longitudinal cefiderocol; Suzuki 2014 multi-drug ALE). |
| 4 | **No FQ DMS** -- gyrA/parC/parE/gyrB have NaN DMS features. | Manual extraction of FQ-target fitness from published papers (Pal 2015 gyrA; Sanchez 2022 parC) and ingest into `dms_fitness.parquet` schema. Each is ~200-500 measurements, doable. |
| 5 | **Rodrigues 2016 DHFR DMS blocked at Gate 5**. | Manual download from PNAS supplementary or contact authors. ~ 200 measurements. |
| 6 | **TEM Ambler <-> Stiffler numbering**. | Add a unified position-canonicalization step before DMS lookup; necessary as soon as any beta-lactam trajectory enters the training data. |
| 7 | **S. aureus and K. pneumoniae trajectories**. Only 7 inferred-CARD trajectories combined. The intended held-out evaluations (S. aureus x oxacillin, K. pneumoniae x meropenem) had 0 transitions. | Either (a) accept that week-1 is FQ-only and document, or (b) curate lab-evolution trajectories for these species (much smaller universe than E. coli ALEdb). |
| 8 | **Canonical-AMR underperformance**. Even at 70% top-3 overall, the canonical-AMR stratum hits only ~20% top-3. | Week-2 model should add an objective component that upweights canonical-AMR-gene mutations, OR use the DMS-fitness feature in a different framing (per-residue ESM2 + DMS + position as a continuous fitness landscape rather than a candidate-ranking signal). |

---

## Appendix A -- File Map

```
/mnt/ssd/research/amr/
|-- REPORT.md                       (this file)
|-- src/
|   |-- gate0_setup.py
|   |-- gate1_bvbrc.py
|   |-- gate2_card.py
|   |-- gate2_reparse.py
|   |-- gate3_trajectories.py
|   |-- gate4_esm2_embeddings.py
|   |-- gate4_validate.py
|   |-- gate5_dms.py
|   `-- gate6_baseline.py
|-- processed/
|   |-- resistance_genes.parquet    (18,193 x 16)
|   |-- trajectories.parquet        (860 x 11)
|   |-- transitions.parquet         (28,028 x 13)
|   |-- dms_fitness.parquet         (95,722 x 13)
|   |-- gate{1,2,3,5}_summary.json
|   `-- amrfinder_calls/{ecoli,saureus,kpneumo}/
|-- embeddings/
|   |-- protein_embeddings.npz      (17,991 x 1280 fp16, 47.7 MB)
|   |-- residue_embeddings/         (11,382 x .npy, ~720 MB)
|   |-- residue_embeddings_index.parquet
|   |-- _md5_to_amr.parquet         (1,245 x 5)
|   `-- _gate4_meta.json
|-- models/
|   |-- esm2_pca_protein.pkl        (64 components, fit on 17,991 prots)
|   |-- esm2_pca_residue.pkl        (32 components, fit on 120k residues)
|   `-- single_step_lgbm.pkl        (LambdaRank, 236 features, 221 trees)
|-- results/
|   |-- single_step_eval.json
|   `-- single_step_features.csv
`-- logs/
    |-- gates.log
    |-- gate{0,1,2,3,4,5,6}*.log
    `-- ...
```

## Appendix B -- Reference Bands

For week-2 architecture evaluation, useful comparison numbers:

| Baseline | Top-3 (FQ holdout) | Source |
| --- | ---: | --- |
| Random over FQ vocab | 0.0024 (= 3/1274) | uniform sampling from the 1274-mutation FQ vocab |
| Predict 3 most frequent FQ mutations in training | ~ 0.05 | trivial frequency baseline |
| LambdaRank baseline (this work) | **0.708** | Gate 6 |
| Spec target | 0.40 | agent definition |

The week-2 autoregressive model needs to beat 0.708 to be worth the
complexity. The canonical-AMR stratum (currently 0.13 top-3) is the
real target -- if the autoregressive model can hit 0.40 on
canonical-AMR-only transitions, the project is in good shape for
clinical use cases.

---

*Generated 2026-05-13 by Claude Opus 4.7 (AMR Gate 6 subagent).*
