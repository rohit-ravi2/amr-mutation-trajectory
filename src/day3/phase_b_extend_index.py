"""Helper: build a NEW residue_embeddings_index.parquet that includes
both short (in residue_embeddings/) and long (in residue_embeddings_long/)
entries, regardless of whether Phase B has fully finished.

This lets Phase D start using partial long-protein embeddings.
"""
from pathlib import Path
import numpy as np
import pandas as pd

EMB = Path('/mnt/ssd/research/amr/embeddings')

# Existing index (short)
idx_short = pd.read_parquet(EMB / 'residue_embeddings_index.parquet')
if 'embedding_method' not in idx_short.columns:
    idx_short['embedding_method'] = 'single_window'

# Scan long dir
long_dir = EMB / 'residue_embeddings_long'
long_files = sorted(long_dir.glob('*.npy'))
print(f'Long-protein files: {len(long_files)}')

rows_long = []
for fp in long_files:
    md5 = fp.stem
    try:
        arr = np.load(fp, mmap_mode='r')
        sl = int(arr.shape[0])
        rows_long.append({'md5': md5, 'seq_len': sl, 'embedding_method': 'chunked_window'})
    except Exception as e:
        print(f'  fail {md5}: {e}')

if rows_long:
    long_df = pd.DataFrame(rows_long)
    print(f'Long index entries to add: {len(long_df)}')
    print(f'  seq_len min={long_df["seq_len"].min()}, max={long_df["seq_len"].max()}, median={long_df["seq_len"].median()}')

    merged = pd.concat([idx_short, long_df], ignore_index=True).drop_duplicates(subset=['md5'])
    merged.to_parquet(EMB / 'residue_embeddings_index.parquet', index=False)
    print(f'Updated index rows: {len(merged)} (was {len(idx_short)})')
else:
    print('No long-protein files yet.')
