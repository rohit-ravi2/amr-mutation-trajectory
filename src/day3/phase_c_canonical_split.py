"""Phase C Day-3: Canonical-AMR-rich eval split.

Build a new eval split where canonical-AMR transitions are *enriched*:
- Train: lab transitions from ALEdb / Baym / etc., EXCLUDING the held-out test set
- Test: clinical / CARD-derived / curated canonical-AMR-rich trajectories
        (Imamovic, Barbosa, Melnyk, PATRIC, CARD_snp_*)

Save splits at /mnt/ssd/research/amr/processed/splits/canonical_amr_split.pkl
"""
import os, sys, json, pickle
from pathlib import Path
import pandas as pd
import numpy as np

SSD = Path('/mnt/ssd/research/amr')
OUT_DIR = SSD / 'processed' / 'splits'
OUT_DIR.mkdir(parents=True, exist_ok=True)

t = pd.read_parquet(SSD / 'processed' / 'transitions.parquet')
print(f'Total transitions: {len(t)}')

# Test sources: clinical, CARD-derived, curated published trajectories
test_source_patterns = [
    'CARD_snp_',
    'Imamovic',
    'Barbosa',
    'Melnyk',
    'PATRIC',
    'inferred_card',
]
pat = '|'.join(test_source_patterns)
test_mask = t['source_paper'].astype(str).str.contains(pat, regex=True, na=False)
test = t[test_mask].copy()
train = t[~test_mask].copy()

print(f'Test transitions (clinical/curated): {len(test)}')
print(f'  canonical_AMR among test: {test["mutation_in_canonical_amr_gene"].sum()}')
print(f'  unique drug_classes in test: {test["drug_class"].nunique()}')
print(f'Train transitions (lab ALEdb / Baym): {len(train)}')
print(f'  canonical_AMR among train: {train["mutation_in_canonical_amr_gene"].sum()}')

# Verify: no trajectory_id leakage (each trajectory must be wholly in one split)
test_trajs = set(test['trajectory_id'])
train_trajs = set(train['trajectory_id'])
overlap = test_trajs & train_trajs
print(f'trajectory_id overlap (must be 0): {len(overlap)}')
assert len(overlap) == 0, 'Trajectory leakage!'

# Also build a per-drug-class breakdown for test
print('\nTest drug_class breakdown:')
print(test['drug_class'].value_counts())

# Save split
split = {
    'train_transition_ids': set(train['transition_id'].astype(str)),
    'test_transition_ids': set(test['transition_id'].astype(str)),
    'test_canonical_n': int(test['mutation_in_canonical_amr_gene'].sum()),
    'description': 'Day-3 canonical-AMR-rich split: TEST=clinical/curated (CARD,Imamovic,Barbosa,Melnyk,PATRIC,inferred); TRAIN=lab ALEdb/Baym/Toprak',
}
with open(OUT_DIR / 'canonical_amr_split.pkl', 'wb') as f:
    pickle.dump(split, f)

# Also save as parquet for inspection
test[['transition_id','trajectory_id','from_genotype','to_genotype','mutation_added',
      'drug','drug_class','step_number','source_paper','lab_or_clinical',
      'mutation_in_canonical_amr_gene','is_hypermutator_trajectory']].to_parquet(
    OUT_DIR / 'canonical_amr_test_transitions.parquet', index=False)
train[['transition_id','trajectory_id','source_paper','mutation_in_canonical_amr_gene','drug_class']].to_parquet(
    OUT_DIR / 'canonical_amr_train_transitions.parquet', index=False)

summary = {
    'n_train': len(train),
    'n_test': len(test),
    'n_test_canonical': int(test['mutation_in_canonical_amr_gene'].sum()),
    'n_train_canonical': int(train['mutation_in_canonical_amr_gene'].sum()),
    'test_drug_class_counts': test['drug_class'].value_counts().to_dict(),
    'train_drug_class_counts': train['drug_class'].value_counts().to_dict(),
    'test_source_paper_counts': test['source_paper'].value_counts().to_dict(),
    'meets_C2_target_>=100_canonical': bool(test['mutation_in_canonical_amr_gene'].sum() >= 100),
}
with open(SSD / 'processed' / 'day3' / 'phase_c_split_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f'\nC2 target met (≥100 canonical-AMR test transitions): {summary["meets_C2_target_>=100_canonical"]}')
print(json.dumps({'n_train': summary['n_train'], 'n_test': summary['n_test'],
                  'n_test_canonical': summary['n_test_canonical']}, indent=2))
