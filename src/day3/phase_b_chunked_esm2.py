"""Phase B Day-3: ESM2-650M chunked-window per-residue embeddings for proteins >512 aa.

For each protein with seq_len > 512:
  - Tile windows of 512 aa with stride 256 (50% overlap).
  - Run ESM2-650M fp16 on each window, extract per-residue from layer 33.
  - For overlapping positions: weighted average across windows containing the position.
    * Non-edge positions (>=64 from window start/end) weight = 1.0
    * Edge positions (first/last 64 aa of a window) weight = 0.5
  - Apply Day-2 learned projector (1280 -> 128) to produce float16 [seq_len, 128].
  - Save to work/embeddings/residue_embeddings_long/<md5>.npy
  - Also: append mean-pooled embedding (post-projector? or pre? we'll save pre-projector
    1280-dim mean for protein_embeddings.npz compatibility)
  - Update residue_embeddings_index.parquet with seq_len and embedding_method="chunked_window".

For Phase B4 sanity: also test single-window vs chunked on TEM-1 1024-aa synthetic duplicate.
"""
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SSD = Path('work')
EMB_DIR = SSD / 'embeddings'
LONG_DIR = EMB_DIR / 'residue_embeddings_long'
LONG_DIR.mkdir(parents=True, exist_ok=True)
OUT = SSD / 'processed' / 'day3'
OUT.mkdir(parents=True, exist_ok=True)

# Day-2 projector model
import json
PROJ_CFG = json.loads((SSD / 'models' / 'residue_projector_cfg.json').read_text())

class ResidueEncoder(nn.Module):
    """Matches Day-2 v2 encoder: 1280 -> 1024 (LN+GELU) -> 512 (LN+GELU) -> 128."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1280, 1024),    # 0
            nn.LayerNorm(1024),       # 1
            nn.GELU(),                 # 2 (parameterless)
            nn.Linear(1024, 512),     # 3
            nn.LayerNorm(512),        # 4
            nn.GELU(),                 # 5 (parameterless)
            nn.Linear(512, 128),      # 6
        )

    def encode(self, x):
        return self.encoder(x)

def load_projector():
    sd = torch.load(SSD / 'models' / 'residue_projector.pt', map_location='cpu', weights_only=True)
    m = ResidueEncoder()
    m.encoder.load_state_dict(sd['encoder_state_dict'])
    m.eval()
    return m.to(DEV)

# ESM2 model
import esm
print('[B] Loading ESM2-650M...', flush=True)
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.half().to(DEV)
model.eval()
batch_converter = alphabet.get_batch_converter()
PAD_IDX = alphabet.padding_idx
LAYER = 33  # last layer for 650M
EMBED_DIM = 1280
print('[B] ESM2 ready.', flush=True)

projector = load_projector()
print('[B] Projector loaded.', flush=True)

# Chunked-window inference
WINDOW = 512
STRIDE = 256  # 50% overlap
EDGE = 64     # edge zone weight 0.5

@torch.no_grad()
def embed_window_batch(seqs):
    """seqs: list of (name, seq) tuples. Returns list of [seq_len, 1280] np arrays (float32 on cpu)."""
    if not seqs:
        return []
    labels, strs, tokens = batch_converter(seqs)
    tokens = tokens.to(DEV)
    out = model(tokens, repr_layers=[LAYER], return_contacts=False)
    reps = out['representations'][LAYER]  # [B, L+2, 1280] fp16
    results = []
    for i, (name, s) in enumerate(seqs):
        L = len(s)
        # strip <bos> and <eos>; bos is at 0, eos at L+1
        r = reps[i, 1:L+1].float().cpu().numpy()
        results.append(r)
    return results

def chunked_residue_embedding(seq, batch_size=8):
    """Returns float32 [len(seq), 1280] using overlapping 512-aa windows."""
    L = len(seq)
    if L <= WINDOW:
        # Single pass
        embs = embed_window_batch([('seq', seq)])
        return embs[0]
    # Build window ranges
    windows = []
    start = 0
    while start < L:
        end = min(start + WINDOW, L)
        windows.append((start, end))
        if end == L:
            break
        start += STRIDE
    # Accumulators
    acc = np.zeros((L, EMBED_DIM), dtype=np.float32)
    wsum = np.zeros((L,), dtype=np.float32)
    # Batch the windows
    for bs in range(0, len(windows), batch_size):
        batch = windows[bs:bs+batch_size]
        seqs = [(f'w{i}', seq[s:e]) for i,(s,e) in enumerate(batch)]
        outs = embed_window_batch(seqs)
        for (s,e), r in zip(batch, outs):
            wlen = e - s
            # Weight per position within window
            w = np.ones(wlen, dtype=np.float32)
            # edge zones at start of window: first EDGE positions (unless this is the very first window of the protein → not edge at protein start)
            if s != 0:
                w[:min(EDGE, wlen)] *= 0.5
            if e != L:
                w[-min(EDGE, wlen):] *= 0.5
            acc[s:e] += r * w[:, None]
            wsum[s:e] += w
    # Guard against zero weights
    wsum = np.maximum(wsum, 1e-6)
    out = acc / wsum[:, None]
    return out

@torch.no_grad()
def project_to_128(residues_1280):
    """residues_1280: np array [L, 1280] float32 -> np array [L, 128] float16."""
    x = torch.tensor(residues_1280, dtype=torch.float32, device=DEV)
    z = projector.encode(x)
    return z.cpu().numpy().astype(np.float16)

# ---------------------- Main job ----------------------
def main():
    # Phase B4 sanity check FIRST (on TEM-1 single vs synthetic-double window)
    print('[B4] Sanity check: TEM-1 single vs synthetic-1024 chunked', flush=True)
    # Use first TEM-1 sequence from md5_to_amr
    m2a = pd.read_parquet(EMB_DIR / '_md5_to_amr.parquet')
    tem_rows = m2a[m2a['Element symbol'].astype(str).str.contains('TEM-1', case=False, regex=False, na=False)]
    tem1_md5 = tem_rows.iloc[0]['Protein id'] if len(tem_rows) else None
    # find sequence
    tem1_seq = None
    if tem1_md5:
        for sp in ['ecoli', 'saureus', 'kpneumo']:
            try:
                p = pd.read_parquet(f'work/raw/proteins_{sp}.parquet')
                hit = p[p['aa_sequence_md5'] == tem1_md5]
                if len(hit):
                    tem1_seq = hit.iloc[0]['aa_sequence']
                    break
            except: pass
    if tem1_seq is None:
        # Fallback: synthesize from a known short protein in residue dir
        idx = pd.read_parquet(EMB_DIR / 'residue_embeddings_index.parquet')
        # pick a ~286 aa protein
        cand = idx[(idx['seq_len']>=280) & (idx['seq_len']<=300)].iloc[0]
        tem1_md5 = cand['md5']
        # load residue embedding directly
        r_single = np.load(EMB_DIR / 'residue_embeddings' / f'{tem1_md5}.npy').astype(np.float32)
        # synthesize: load original sequence by finding md5 -> need sequence; just compare to disk-stored
        print(f'  fallback: using existing residue file {tem1_md5} seq_len={r_single.shape[0]}')
        # Synthetic: duplicate to mimic 2*L protein
        # But we have no original sequence to feed. Skip synthetic chunked test in fallback.
        cosine_check = None
        # Instead: chunk a longer canonical-long protein and report internal consistency
    else:
        L = len(tem1_seq)
        print(f'  TEM-1 found: md5={tem1_md5[:10]}..., seq_len={L}')
        # Single window pass
        r_single = embed_window_batch([('tem1', tem1_seq)])[0]
        # Synthetic 1024-aa duplicate: tem1_seq + tem1_seq
        synth = tem1_seq + tem1_seq
        r_chunk = chunked_residue_embedding(synth, batch_size=4)
        # Compare TEM-1 positions in the first half of synth to single
        # cosine per position
        a = r_single
        b = r_chunk[:L]  # first L positions of synthetic duplicate
        # Renormalize
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        cos = (a_n * b_n).sum(axis=1)
        cosine_check = {
            'tem1_md5': tem1_md5,
            'tem1_len': int(L),
            'mean_cosine_first_half': float(cos.mean()),
            'median_cosine_first_half': float(np.median(cos)),
            'min_cosine_first_half': float(cos.min()),
            'pct_above_0.95': float((cos > 0.95).mean()),
        }
        print(f'  TEM-1 chunked-vs-single mean cosine = {cos.mean():.4f}, pct>0.95 = {(cos>0.95).mean():.4f}')
        # Also check 2nd half (positions L..2L) cosines back to single — should also be high since identical seq
        b2 = r_chunk[L:2*L]
        b2_n = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
        cos2 = (a_n * b2_n).sum(axis=1)
        cosine_check['mean_cosine_second_half'] = float(cos2.mean())

    # Save sanity result
    with open(OUT / 'phase_b_sanity.json', 'w') as f:
        json.dump(cosine_check, f, indent=2, default=str)
    print(f'[B4] sanity saved: {cosine_check}', flush=True)

    # ------- Main: embed all NEW canonical-long proteins (Phase B1-B3) -------
    new_long = pd.read_parquet(OUT / 'phase_a_new_canonical_long_proteins.parquet')
    # Also include NEW long AMR-keyword hits (RND efflux pumps etc.)
    new_amr = pd.read_parquet(OUT / 'phase_a_new_amr_proteins.parquet')
    new_amr_long = new_amr[new_amr['aa_length'] > 512].copy()
    # Combine
    cols = ['aa_sequence_md5','aa_sequence','aa_length','gene','product']
    new_long_set = pd.concat([new_long[cols], new_amr_long[cols]], ignore_index=True)
    new_long_set = new_long_set.drop_duplicates(subset=['aa_sequence_md5'])
    # Filter out any already embedded
    existing_long = set(p.stem for p in LONG_DIR.glob('*.npy'))
    new_long_set = new_long_set[~new_long_set['aa_sequence_md5'].isin(existing_long)]
    # Sort shortest first (faster iteration / progress)
    new_long_set = new_long_set.sort_values('aa_length').reset_index(drop=True)
    print(f'[B1] Long proteins to embed: {len(new_long_set)}')
    print(f'[B1] Length range: {new_long_set["aa_length"].min()}-{new_long_set["aa_length"].max()}')

    # Append mean-pooled to protein_embeddings.npz at the end via a dict accumulator
    existing_npz = np.load(EMB_DIR / 'protein_embeddings.npz')
    new_pooled = {}
    long_index_rows = []

    t0 = time.time()
    n_done = 0
    pbar = tqdm(new_long_set.iterrows(), total=len(new_long_set), desc='chunked-embed')
    for _, row in pbar:
        md5 = row['aa_sequence_md5']
        seq = str(row['aa_sequence'])
        L = len(seq)
        try:
            r1280 = chunked_residue_embedding(seq, batch_size=8)
            r128 = project_to_128(r1280)
            # Save residue (projected) to long dir
            np.save(LONG_DIR / f'{md5}.npy', r128)
            # Mean-pool 1280 to add to protein_embeddings.npz
            mean_pool = r1280.mean(axis=0).astype(np.float16)
            new_pooled[md5] = mean_pool
            long_index_rows.append({'md5': md5, 'seq_len': int(L), 'embedding_method': 'chunked_window'})
            n_done += 1
            # checkpoint every 200
            if n_done % 200 == 0:
                # save intermediate index
                idx_df = pd.DataFrame(long_index_rows)
                idx_df.to_parquet(OUT / 'phase_b_long_index_partial.parquet', index=False)
                elapsed = time.time() - t0
                rate = n_done / elapsed
                pbar.set_postfix(rate=f'{rate:.2f}/s', n=n_done)
        except Exception as e:
            print(f'  [B] FAIL md5={md5[:10]} L={L} err={e}')
            torch.cuda.empty_cache()

    print(f'[B] Long proteins embedded: {n_done}/{len(new_long_set)}', flush=True)
    print(f'[B] Wall: {(time.time()-t0)/60:.1f} min', flush=True)

    # --- Append new mean-pooled to protein_embeddings.npz ---
    # Merge: existing + new
    existing_dict = {k: existing_npz[k] for k in existing_npz.files}
    existing_dict.update(new_pooled)
    np.savez(EMB_DIR / 'protein_embeddings.npz', **existing_dict)
    print(f'[B] protein_embeddings.npz updated: {len(existing_dict)} total', flush=True)

    # --- Update residue_embeddings_index.parquet ---
    idx_path = EMB_DIR / 'residue_embeddings_index.parquet'
    idx_old = pd.read_parquet(idx_path)
    if 'embedding_method' not in idx_old.columns:
        idx_old['embedding_method'] = 'single_window'
    idx_long = pd.DataFrame(long_index_rows)
    idx_new = pd.concat([idx_old, idx_long], ignore_index=True).drop_duplicates(subset=['md5'])
    idx_new.to_parquet(idx_path, index=False)
    print(f'[B] residue_embeddings_index.parquet updated: {len(idx_new)} rows', flush=True)

    # Final summary
    summary = {
        'sanity': cosine_check,
        'n_long_embedded': n_done,
        'n_attempted': int(len(new_long_set)),
        'wall_min': float((time.time()-t0)/60),
        'total_pooled_embeddings_now': len(existing_dict),
        'total_residue_index_rows_now': int(len(idx_new)),
    }
    with open(OUT / 'phase_b_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))

if __name__ == '__main__':
    main()
