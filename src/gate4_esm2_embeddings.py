#!/usr/bin/env python
"""
Gate 4 — ESM2-650M embedding precomputation for AMR proteins.

Strategy:
- Build unique sequence set across the three species protein parquets (md5-keyed).
- Build priority tiers:
   1) Sequences from genes appearing in transitions.parquet
   2) Resistance proteins (md5 -> amr_class major family) from AMRFinder calls
   3) Everything else
- Embed in priority order with ESM2-650M (fp16, GPU). Mean-pool last layer -> [1280].
- Save residue-level embeddings for sequences <= 512 aa.
- Resumable: done_md5s.txt + per-batch flush to per-md5 .npy files; final repack to npz.
- Validate via TEM-1 / TEM-52 cosine similarity (must be in [0.70, 0.99]).

Outputs:
  work/embeddings/
    protein_embeddings.npz                       # md5 -> [1280] float16
    pooled/<md5>.npy                             # per-md5 pooled float16 [1280] (intermediate)
    residue_embeddings/<md5>.npy                 # per-md5 residue float16 [L,1280] for L<=512
    residue_embeddings_index.parquet             # (md5, seq_len)
    checkpoints/checkpoint_<N>.npz               # rolling pooled checkpoints
    done_md5s.txt                                # resumability log
    _md5_to_amr.parquet                          # md5 -> amr_class/symbol mapping
    _gate4_meta.json
"""

import os
import sys
import json
import time
import glob
import math
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------ paths and logging ------------------

EMB_DIR = Path('work/embeddings')
POOLED_DIR = EMB_DIR / 'pooled'
RES_DIR = EMB_DIR / 'residue_embeddings'
CHK_DIR = EMB_DIR / 'checkpoints'
LOG_PATH = Path('work/logs/gate4.log')
GATES_LOG = Path('work/logs/gates.log')

EMB_DIR.mkdir(parents=True, exist_ok=True)
POOLED_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
CHK_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)sZ %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger('gate4')

DONE_PATH = EMB_DIR / 'done_md5s.txt'

# ------------------ utilities ------------------

def notify(msg, title='AMR Gate 4', priority=None):
    try:
        cmd = [os.environ.get('NOTIFY_BIN', 'notify'), msg, title]
        if priority:
            cmd.append(priority)
        subprocess.run(cmd, check=False, timeout=10)
    except Exception:
        pass

def md5_of(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()

def load_done():
    if DONE_PATH.exists():
        with open(DONE_PATH) as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def mark_done(md5s):
    with open(DONE_PATH, 'a') as f:
        for m in md5s:
            f.write(m + '\n')

# ------------------ sequence loading ------------------

def build_sequence_set():
    """Return dict md5 -> sequence with priority tier."""
    log.info("Loading proteins parquets...")
    all_prots = []
    for sp in ['ecoli', 'saureus', 'kpneumo']:
        df = pd.read_parquet(f'work/raw/proteins_{sp}.parquet',
                             columns=['aa_sequence_md5', 'aa_sequence', 'aa_length', 'product'])
        df['species'] = sp
        all_prots.append(df)
    prots = pd.concat(all_prots, ignore_index=True)
    log.info(f"Total protein rows across species: {len(prots)}")

    # Deduplicate on md5
    uniq = prots.drop_duplicates(subset='aa_sequence_md5').dropna(subset=['aa_sequence_md5', 'aa_sequence'])
    log.info(f"Unique md5 sequences: {len(uniq)}")

    # Filter out empty/stop-only sequences
    uniq = uniq[uniq['aa_sequence'].str.len() >= 5].copy()
    log.info(f"After length filter (>=5 aa): {len(uniq)}")

    # Priority tier 2: resistance md5s (from AMRFinder calls)
    md5_amr = pd.read_parquet(EMB_DIR / '_md5_to_amr.parquet')
    major_classes = {'BETA-LACTAM', 'QUINOLONE', 'AMINOGLYCOSIDE', 'TRIMETHOPRIM',
                     'SULFONAMIDE', 'TETRACYCLINE'}
    major_md5 = set(md5_amr[md5_amr['Class'].isin(major_classes)]['Protein id'])
    all_resistance_md5 = set(md5_amr['Protein id'])

    # Priority tier 1: sequences from genes appearing in transitions
    # Map transition gene-symbols to md5s.
    # transitions.mutation_added is like 'gyrA_S83L'. The base gene is split('_')[0].
    # In AMRFinder Element symbol we have e.g. 'gyrA_S83L' or 'blaCMY-2'. We can match base of element symbol.
    tr = pd.read_parquet('work/processed/transitions.parquet',
                         columns=['mutation_added'])
    tr_genes_base = set(tr['mutation_added'].dropna().str.split('_').str[0].str.lower())
    log.info(f"Transition base-gene tokens: {len(tr_genes_base)}")

    md5_amr['_sym_base'] = md5_amr['Element symbol'].str.split('_').str[0].str.lower()
    transition_md5 = set(md5_amr[md5_amr['_sym_base'].isin(tr_genes_base)]['Protein id'])
    log.info(f"AMRFinder md5s linked to transition genes: {len(transition_md5)}")

    # Assign priority
    def prio(md5):
        if md5 in transition_md5:
            return 1
        if md5 in major_md5:
            return 2
        if md5 in all_resistance_md5:
            return 3
        return 4

    uniq['_prio'] = uniq['aa_sequence_md5'].map(prio)
    uniq = uniq.sort_values(['_prio', 'aa_length'], ascending=[True, True]).reset_index(drop=True)
    log.info("Priority tier counts: " + str(uniq['_prio'].value_counts().sort_index().to_dict()))

    return uniq

# ------------------ ESM2 embedding ------------------

def choose_batch_size():
    import torch
    if not torch.cuda.is_available():
        return 1, 'cpu'
    free_bytes, _ = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1e9
    if free_gb >= 5.0:
        return 8, 'cuda'
    if free_gb >= 4.0:
        return 4, 'cuda'
    if free_gb >= 3.0:
        return 2, 'cuda'
    if free_gb >= 1.5:
        return 1, 'cuda'
    return 1, 'cpu'

MAX_LEN = 1022  # ESM2 token budget: bos + L + eos <= 1024

def chunk_long(seq, window=MAX_LEN, stride=512):
    """Split a long sequence into overlapping windows for pooling."""
    if len(seq) <= window:
        return [(0, seq)]
    chunks = []
    start = 0
    while start < len(seq):
        end = min(start + window, len(seq))
        chunks.append((start, seq[start:end]))
        if end == len(seq):
            break
        start += stride
    return chunks

def embed_batch(model, alphabet, batch_converter, device, batch_seqs):
    """batch_seqs: list of (md5_or_label, sub_seq). All sub_seqs same-ish length OK; use padding."""
    import torch
    labels = [s[0] for s in batch_seqs]
    seqs = [s[1] for s in batch_seqs]
    data = list(zip(labels, seqs))
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[33], return_contacts=False)
    reps = out['representations'][33]  # [B, L+2, 1280]
    # Strip BOS/EOS and pad. Use alphabet padding for proper mask.
    pad_idx = alphabet.padding_idx
    mask = (tokens != pad_idx) & (tokens != alphabet.cls_idx) & (tokens != alphabet.eos_idx)
    # Note: alphabet.cls_idx / eos_idx
    return reps.cpu(), tokens.cpu(), mask.cpu()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Embed at most this many (after resume). For testing.')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only run TEM-1/TEM-52 validation.')
    args = parser.parse_args()

    import torch
    notify("Gate 4 starting: ESM2-650M embedding precomputation")

    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        log.info(f"GPU: {torch.cuda.get_device_name(0)} | free={free_bytes/1e9:.2f} GB / total={total_bytes/1e9:.2f} GB")
    else:
        log.warning("CUDA NOT available — will fall back to CPU. Very slow.")

    batch_size, device_kind = choose_batch_size()
    log.info(f"Chosen batch size: {batch_size} on {device_kind}")

    # ------------------ load model ------------------
    log.info("Loading ESM2-t33-650M...")
    from esm import pretrained
    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    if device_kind == 'cuda':
        try:
            model = model.half().cuda()
        except torch.cuda.OutOfMemoryError:
            log.warning("OOM during model.half().cuda(); falling back to CPU.")
            device_kind = 'cpu'
            batch_size = 1
            model = model.float().cpu()
    else:
        model = model.float().cpu()
    model.eval()
    device = next(model.parameters()).device
    log.info(f"Model on device: {device}")
    notify("ESM2-650M loaded successfully")

    # ------------------ TEM-1 / TEM-52 validation ------------------
    TEM1 = ("MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSR"
            "VDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDR"
            "WEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGI"
            "IAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW")
    TEM52 = ("MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSR"
             "VDAGQEQLGRRIHYSQNDLVKYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDR"
             "WEPELNEAIPNDERDTTTPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGASERGSRGI"
             "IAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW")
    log.info(f"TEM-1 len={len(TEM1)}, TEM-52 len={len(TEM52)}")

    def embed_one(seq):
        _, _, tokens = batch_converter([("x", seq)])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[33], return_contacts=False)
        rep = out['representations'][33][0]  # [L+2, 1280]
        # strip BOS/EOS
        rep = rep[1:1 + len(seq)]
        pooled = rep.mean(dim=0).float().cpu().numpy().astype(np.float16)
        residue = rep.float().cpu().numpy().astype(np.float16)
        return pooled, residue

    tem1_vec, _ = embed_one(TEM1)
    tem52_vec, _ = embed_one(TEM52)
    cos = float(np.dot(tem1_vec.astype(np.float32), tem52_vec.astype(np.float32)) /
                (np.linalg.norm(tem1_vec.astype(np.float32)) * np.linalg.norm(tem52_vec.astype(np.float32))))
    log.info(f"TEM-1/TEM-52 cosine similarity: {cos:.4f}")
    notify(f"TEM-1/TEM-52 cosine = {cos:.4f}")
    tem_pass = 0.70 <= cos <= 0.99
    log.info(f"TEM validation: {'PASS' if tem_pass else 'FAIL'}")

    if args.validate_only:
        log.info("validate-only: exiting after TEM check.")
        return cos, tem_pass

    # ------------------ build work list ------------------
    uniq = build_sequence_set()
    done = load_done()
    log.info(f"Already done from prior runs: {len(done)}")
    todo = uniq[~uniq['aa_sequence_md5'].isin(done)].copy()
    log.info(f"Remaining to embed: {len(todo)}")

    if args.max_sequences is not None:
        todo = todo.head(args.max_sequences)
        log.info(f"Limit to first {len(todo)} for this run")

    # ------------------ embed loop ------------------
    t0 = time.time()
    rows_residue_index = []
    batch_buf = []   # list of (md5, full_seq, sub_seq, chunk_idx, total_chunks, chunk_start)
    pooled_acc = {}  # md5 -> list of (chunk_pooled, chunk_len)

    # Build per-row work units: for short seqs one unit; for long seqs multiple chunks
    work_units = []
    for _, row in todo.iterrows():
        md5 = row['aa_sequence_md5']
        seq = row['aa_sequence']
        seq = seq.replace('*', '')  # strip stop codon if present
        if len(seq) == 0:
            continue
        chunks = chunk_long(seq, window=MAX_LEN, stride=512)
        for ci, (cs, sub) in enumerate(chunks):
            work_units.append((md5, len(seq), ci, len(chunks), cs, sub))
    log.info(f"Total chunk work-units: {len(work_units)}")

    # Group by similar length to minimize padding waste; within batch all <= MAX_LEN+2 tokens
    # Sort by sub-seq length to bucket similar
    work_units.sort(key=lambda u: len(u[5]))

    # We'll process in batches sized by current batch_size, halving on OOM.
    cur_bs = batch_size
    n_emb = 0
    n_since_chk = 0
    chk_buf = {}  # md5 -> pooled
    n_since_progress = 0
    progress_step = 1000

    i = 0
    pbar = tqdm(total=len(work_units), desc="Embedding chunks")
    while i < len(work_units):
        batch = work_units[i:i + cur_bs]
        # Build batch_converter input
        data = [(f"{u[0]}_{u[2]}", u[5]) for u in batch]
        try:
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[33], return_contacts=False)
            reps = out['representations'][33]  # [B, L+2, 1280]
            # For each sequence in batch, strip BOS/EOS and the padding
            for bi, u in enumerate(batch):
                md5, full_len, ci, total_ch, cs, sub = u
                L = len(sub)
                rep = reps[bi, 1:1 + L]  # [L, 1280]
                pooled = rep.mean(dim=0).float().cpu().numpy().astype(np.float16)
                pooled_acc.setdefault(md5, []).append((pooled, L, ci, total_ch))
                # If sequence short and single chunk, save residue
                if full_len <= 512 and total_ch == 1:
                    residue = rep.float().cpu().numpy().astype(np.float16)
                    np.save(RES_DIR / f"{md5}.npy", residue)
                    rows_residue_index.append({'md5': md5, 'seq_len': L})
            i += len(batch)
            pbar.update(len(batch))

            # When all chunks of an md5 are collected, finalize the pooled vector
            # We need to know how many chunks per md5; finalize lazily
            finalized = []
            for md5, items in list(pooled_acc.items()):
                # total_ch from any item
                total_ch = items[0][3]
                if len(items) == total_ch:
                    # length-weighted mean pool across chunks
                    weights = np.array([it[1] for it in items], dtype=np.float32)
                    vecs = np.stack([it[0].astype(np.float32) for it in items], axis=0)
                    final = (vecs * weights[:, None]).sum(axis=0) / weights.sum()
                    final = final.astype(np.float16)
                    np.save(POOLED_DIR / f"{md5}.npy", final)
                    chk_buf[md5] = final
                    finalized.append(md5)
            for m in finalized:
                pooled_acc.pop(m, None)

            if finalized:
                mark_done(finalized)
                n_emb += len(finalized)
                n_since_chk += len(finalized)
                n_since_progress += len(finalized)

            # Rolling checkpoint
            if n_since_chk >= 500:
                ckp = CHK_DIR / f"checkpoint_{n_emb}.npz"
                np.savez_compressed(ckp, **{k: v for k, v in chk_buf.items()})
                log.info(f"Checkpoint saved: {ckp} ({len(chk_buf)} pooled in buffer)")
                chk_buf = {}
                n_since_chk = 0

            if n_since_progress >= progress_step:
                elapsed = time.time() - t0
                rate = n_emb / max(elapsed, 1e-6)
                log.info(f"Progress: {n_emb} embedded ({rate:.2f} seq/s)")
                notify(f"Gate 4 progress: {n_emb} sequences embedded ({rate:.2f}/s)")
                n_since_progress = 0

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                log.warning(f"OOM — reducing batch size to {cur_bs}")
                continue
            else:
                # Try CPU fallback for this batch
                log.error(f"OOM at batch_size=1, sub_len={len(batch[0][5])}. Skipping this chunk.")
                i += 1
                pbar.update(1)
                continue
        except Exception as e:
            log.exception(f"Error in batch starting at {i}: {e}")
            # Skip this batch
            i += len(batch)
            pbar.update(len(batch))

    pbar.close()

    # Final checkpoint
    if chk_buf:
        ckp = CHK_DIR / f"checkpoint_final.npz"
        np.savez_compressed(ckp, **{k: v for k, v in chk_buf.items()})
        log.info(f"Final checkpoint saved: {ckp}")

    # ------------------ repack final outputs ------------------
    log.info("Repacking pooled .npy files into protein_embeddings.npz ...")
    pooled_files = list(POOLED_DIR.glob('*.npy'))
    log.info(f"Pooled files on disk: {len(pooled_files)}")
    out = {}
    for f in tqdm(pooled_files, desc="repack-pooled"):
        md5 = f.stem
        out[md5] = np.load(f)
    np.savez_compressed(EMB_DIR / 'protein_embeddings.npz', **out)

    # Residue index
    if rows_residue_index:
        residx = pd.DataFrame(rows_residue_index).drop_duplicates(subset='md5')
        residx.to_parquet(EMB_DIR / 'residue_embeddings_index.parquet')
    else:
        # If we resumed and didn't add rows, rebuild from existing files
        files = list(RES_DIR.glob('*.npy'))
        rows = []
        for f in files:
            arr = np.load(f, mmap_mode='r')
            rows.append({'md5': f.stem, 'seq_len': int(arr.shape[0])})
        pd.DataFrame(rows).to_parquet(EMB_DIR / 'residue_embeddings_index.parquet')

    elapsed = time.time() - t0
    log.info(f"Embedding loop wall-clock: {elapsed/60.0:.2f} min")

    return cos, tem_pass


if __name__ == '__main__':
    main()
