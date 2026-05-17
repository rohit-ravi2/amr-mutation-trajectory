"""Phase A continuation: sample-AMRFinder on new genomes (300 per species) to extend
resistance_genes.parquet without running on all 14k new genomes.

Strategy:
  - For each species, sample 300 genomes that are NOT already in amrfinder_calls/<sp>/
    AND that have AMR-keyword-hit proteins in their fastas.
  - Write each genome's protein FASTA to a temp file, run AMRFinder.
  - Parse TSV output, append to existing resistance_genes.parquet.

This is intentionally a sample — the goal is to extend gene_to_md5 coverage for
the new genome batches, not exhaustive annotation.
"""
import os, sys, json, time, subprocess, tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

SSD = Path('/mnt/ssd/research/amr')
RAW = Path('/mnt/hdd/research/amr/raw')
OUT_DIR = SSD / 'processed' / 'amrfinder_calls'
DB = SSD / 'processed' / 'amrfinderplus_db'  # exists from Day-2

# Find DB version dir
db_versions = sorted([p for p in DB.glob('*') if p.is_dir()])
if not db_versions:
    print('ERROR: no amrfinder DB found')
    sys.exit(1)
DB_LATEST = db_versions[-1]
print(f'DB: {DB_LATEST}')

SAMPLES_PER_SPECIES = 300

def write_fasta(df_proteins, out_path):
    """Write {feature_id} as fasta headers and aa_sequence as records."""
    with open(out_path, 'w') as f:
        for _, r in df_proteins.iterrows():
            seq = str(r['aa_sequence']).rstrip('*')
            if not seq:
                continue
            f.write(f'>{r["feature_id"]}\n{seq}\n')

AMRFINDER_BIN = '/home/rohit/miniconda3/envs/amrfinder/bin/amrfinder'

def run_amrfinder(fasta_path, out_path, organism=None):
    """Run AMRFinder protein mode."""
    cmd = [
        AMRFINDER_BIN,
        '-p', str(fasta_path),
        '-d', str(DB_LATEST),
        '-o', str(out_path),
        '--threads', '2',
        '--quiet',
    ]
    if organism:
        cmd.extend(['-O', organism])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return False, res.stderr
    return True, ''

SPECIES_MAP = {
    'ecoli': ('Escherichia', 'ecoli'),
    'saureus': ('Staphylococcus_aureus', 'saureus'),
    'kpneumo': ('Klebsiella_pneumoniae', 'kpneumo'),
}

new_rows = []
for sp, (amrf_org, dirname) in SPECIES_MAP.items():
    print(f'\n[{sp}] starting...')
    sp_dir = OUT_DIR / dirname
    sp_dir.mkdir(parents=True, exist_ok=True)
    done = set(p.stem for p in sp_dir.glob('*.tsv'))
    print(f'  already called: {len(done)}')

    # Load proteins
    p = pd.read_parquet(RAW / f'proteins_{sp}.parquet')
    all_genomes = sorted(p['genome_id'].astype(str).unique())
    new_genomes = [g for g in all_genomes if g not in done]
    print(f'  new genomes: {len(new_genomes)}, sampling {SAMPLES_PER_SPECIES}')

    rng = np.random.default_rng(42)
    sample = list(rng.choice(new_genomes, size=min(SAMPLES_PER_SPECIES, len(new_genomes)), replace=False))

    n_ok = 0
    t0 = time.time()
    pbar = tqdm(sample, desc=f'amrf-{sp}')
    for gid in pbar:
        out_tsv = sp_dir / f'{gid}.tsv'
        if out_tsv.exists():
            n_ok += 1; continue
        sub = p[p['genome_id'].astype(str) == gid]
        if len(sub) == 0:
            continue
        with tempfile.NamedTemporaryFile(mode='w', suffix='.faa', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            write_fasta(sub, tmp_path)
            ok, err = run_amrfinder(tmp_path, out_tsv, organism=amrf_org)
            if ok:
                n_ok += 1
            else:
                print(f'  AMRF fail {gid}: {err[:200]}')
        finally:
            try: os.unlink(tmp_path)
            except: pass
        pbar.set_postfix(ok=n_ok, rate=f'{n_ok/(time.time()-t0+1e-6):.2f}/s')
    print(f'[{sp}] done: {n_ok}/{len(sample)} in {(time.time()-t0)/60:.1f} min')

print('\n[Phase A AMRFinder extend] complete')
