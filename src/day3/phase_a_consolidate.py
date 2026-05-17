"""Phase A consolidation:
1. Merge proteins_<sp>.parquet.parts/* into the main proteins_<sp>.parquet.
2. Identify NEW unique aa_sequence_md5s not yet in protein_embeddings.npz.
3. Save a NEW_SEQS table for downstream embedding + AMRFinder runs.
"""
import os, glob, hashlib, json, time
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path('/mnt/hdd/research/amr/raw')
SSD = Path('/mnt/ssd/research/amr')
EMB = SSD / 'embeddings' / 'protein_embeddings.npz'
OUT = SSD / 'processed' / 'day3'
OUT.mkdir(parents=True, exist_ok=True)

# Load existing embeddings to know what's already done
emb = np.load(EMB)
existing_md5s = set(emb.files)
print(f'[A] existing embeddings: {len(existing_md5s)} md5s')

new_rows = []
all_summary = {}
for sp in ['ecoli', 'saureus', 'kpneumo']:
    base = RAW / f'proteins_{sp}.parquet'
    parts_dir = RAW / f'proteins_{sp}.parquet.parts'
    print(f'\n[{sp}] base={base.exists()} parts_dir={parts_dir.exists()}')

    # Load base
    df_base = pd.read_parquet(base) if base.exists() else pd.DataFrame()
    n_base = len(df_base)
    g_base = df_base['genome_id'].nunique() if n_base else 0

    # Load parts (if any)
    if parts_dir.exists():
        part_files = sorted(glob.glob(str(parts_dir / '*.parquet')))
        if part_files:
            print(f'  reading {len(part_files)} parts...')
            df_parts = pd.concat([pd.read_parquet(f) for f in part_files], ignore_index=True)
            print(f'  parts rows={len(df_parts)} genomes={df_parts["genome_id"].nunique()}')
            df_all = pd.concat([df_base, df_parts], ignore_index=True)
            # dedup by feature_id
            df_all = df_all.drop_duplicates(subset=['feature_id'])
            df_all.to_parquet(base, index=False)
            print(f'  merged: rows={len(df_all)} genomes={df_all["genome_id"].nunique()}')
        else:
            df_all = df_base
    else:
        df_all = df_base

    # Now identify new unique md5s with sequence and aa_length info
    df_all['aa_length'] = df_all['aa_length'].fillna(df_all['aa_sequence'].astype(str).str.len()).astype(int)
    # Keep one row per unique md5 (the first one we see)
    uniq = df_all.drop_duplicates(subset=['aa_sequence_md5'])
    print(f'  unique md5s in {sp}: {len(uniq)}')
    # Filter to NEW (not in existing_md5s)
    new = uniq[~uniq['aa_sequence_md5'].isin(existing_md5s)]
    print(f'  NEW md5s (not in embeddings): {len(new)}')
    new = new[['aa_sequence_md5', 'aa_sequence', 'aa_length', 'gene', 'product', 'genome_id', '_amr_keyword_hit']].copy()
    new['species'] = sp
    new_rows.append(new)

    all_summary[sp] = {
        'rows_total': int(len(df_all)),
        'genomes_total': int(df_all['genome_id'].nunique()),
        'unique_md5s_total': int(len(uniq)),
        'new_unique_md5s': int(len(new)),
    }

new_df = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()
# Dedup across species (a md5 could appear in multiple species)
new_df = new_df.drop_duplicates(subset=['aa_sequence_md5'])
print(f'\n[A] TOTAL new unique md5s (deduplicated across species): {len(new_df)}')

# Save for downstream
new_df.to_parquet(OUT / 'phase_a_new_unique_proteins.parquet', index=False)

# Filter for AMR-relevant: either amr_keyword_hit==True OR aa_length within candidate range
# For ESM2: embed ALL new uniqes (later we'll subset to AMR)
# But that's a lot — let's prioritize AMR keyword hits + GyrA/ParC/RpoB candidates
amr_focus = new_df[new_df['_amr_keyword_hit'] == True].copy()
print(f'[A] new AMR-keyword-hit md5s: {len(amr_focus)}')
amr_focus.to_parquet(OUT / 'phase_a_new_amr_proteins.parquet', index=False)

# Length distribution
print(f'[A] AMR-focus length distribution:')
print(amr_focus['aa_length'].describe())
print(f'    n_gt_512: {(amr_focus["aa_length"]>512).sum()}')

# Also identify the long proteins (>512 aa) for Phase B chunked-window
# These are the gyrA, parC, etc. The 'gene' column should help; product also useful
long_amr = amr_focus[amr_focus['aa_length'] > 512].copy()
print(f'[A] AMR-focus LONG (>512) md5s for Phase B: {len(long_amr)}')

# Also: long proteins from ALL of df_all that include canonical AMR genes (gyrA/parC/rpoB/gyrB/parE) by product text
# These might not have _amr_keyword_hit=True but are critical for QRDR analysis
def is_canonical_long(product, gene):
    s = str(product).lower() + ' ' + str(gene).lower()
    for kw in ['gyra', 'gyrb', 'parc', 'pare', 'rpob', 'mexb', 'acrb', 'dna gyrase', 'topoisomerase iv', 'rna polymerase']:
        if kw in s:
            return True
    return False

# Look across full data for canonical long proteins not in embeddings yet
canonical_long_rows = []
for sp in ['ecoli', 'saureus', 'kpneumo']:
    df = pd.read_parquet(RAW / f'proteins_{sp}.parquet')
    df['_canonical_long'] = df.apply(lambda r: is_canonical_long(r['product'], r['gene']), axis=1)
    cl = df[df['_canonical_long']].drop_duplicates(subset=['aa_sequence_md5'])
    cl = cl[~cl['aa_sequence_md5'].isin(existing_md5s)]
    cl['species'] = sp
    canonical_long_rows.append(cl[['aa_sequence_md5','aa_sequence','aa_length','gene','product','genome_id','species']])
    print(f'  [{sp}] canonical-long-protein NEW md5s: {len(cl)}')

canonical_long_df = pd.concat(canonical_long_rows, ignore_index=True).drop_duplicates(subset=['aa_sequence_md5'])
print(f'[A] TOTAL canonical-long NEW md5s for Phase B: {len(canonical_long_df)}')
canonical_long_df.to_parquet(OUT / 'phase_a_new_canonical_long_proteins.parquet', index=False)

summary = {
    'existing_embedded_md5s': len(existing_md5s),
    'per_species': all_summary,
    'total_new_unique_md5s': int(len(new_df)),
    'new_amr_keyword_md5s': int(len(amr_focus)),
    'new_amr_long_md5s': int((amr_focus['aa_length']>512).sum()),
    'new_canonical_long_md5s': int(len(canonical_long_df)),
}
with open(OUT / 'phase_a_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('\n[A] summary written.')
print(json.dumps(summary, indent=2))
