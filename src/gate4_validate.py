#!/usr/bin/env python
"""Gate 4 validation: TEM-1/TEM-52 + resistance spread + coverage checks."""
import json
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path

EMB_DIR = Path('work/embeddings')

def md5_of(seq):
    return hashlib.md5(seq.encode()).hexdigest()

def cos(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---- Load all embeddings
print("Loading protein_embeddings.npz...")
z = np.load(EMB_DIR / 'protein_embeddings.npz')
md5_keys = list(z.keys())
print(f"Total embedded md5s: {len(md5_keys)}")
embs = {k: z[k] for k in md5_keys}

# ---- Coverage check
print("\n== Coverage vs resistance_genes ==")
md5_amr = pd.read_parquet(EMB_DIR / '_md5_to_amr.parquet')
resistance_md5 = set(md5_amr['Protein id'].dropna())
print(f"Unique resistance md5s in AMRFinder: {len(resistance_md5)}")
covered = resistance_md5 & set(md5_keys)
print(f"Covered by embeddings: {len(covered)} / {len(resistance_md5)} = {len(covered)/max(len(resistance_md5),1)*100:.1f}%")
missing = resistance_md5 - set(md5_keys)
if missing:
    print(f"Missing (first 5): {list(missing)[:5]}")

# ---- TEM-1 / TEM-52 validation
print("\n== TEM-1 / TEM-52 validation ==")
TEM1 = ("MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSR"
        "VDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDR"
        "WEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGI"
        "IAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW")
TEM52 = ("MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSR"
         "VDAGQEQLGRRIHYSQNDLVKYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDR"
         "WEPELNEAIPNDERDTTTPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGASERGSRGI"
         "IAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW")

tem1_md5 = md5_of(TEM1)
tem52_md5 = md5_of(TEM52)
print(f"TEM-1 md5: {tem1_md5}  in npz: {tem1_md5 in embs}")
print(f"TEM-52 md5: {tem52_md5}  in npz: {tem52_md5 in embs}")

# Force re-embed for validation (these reference sequences may not be in BV-BRC genomes)
import torch
from esm import pretrained
print("Loading ESM2 for fresh TEM embedding...")
model, alphabet = pretrained.esm2_t33_650M_UR50D()
model = model.half().cuda()
model.eval()
batch_converter = alphabet.get_batch_converter()

def embed_seq_full(seq):
    _, _, tokens = batch_converter([("x", seq)])
    tokens = tokens.cuda()
    with torch.no_grad():
        out = model(tokens, repr_layers=[33], return_contacts=False)
    rep = out['representations'][33][0][1:1+len(seq)].float().cpu().numpy()
    pooled = rep.mean(0).astype(np.float16)
    return pooled, rep

tem1_pool, tem1_res = embed_seq_full(TEM1)
tem52_pool, tem52_res = embed_seq_full(TEM52)
print(f"TEM-1 / TEM-52 mean-pooled cosine: {cos(tem1_pool, tem52_pool):.4f}")

# Per-position at substitutions
subs = [(i, TEM1[i], TEM52[i]) for i in range(len(TEM1)) if TEM1[i] != TEM52[i]]
print(f"Substitutions: {subs}")
for (i, a1, a52) in subs:
    print(f"  Pos {i+1} ({a1}->{a52}): per-residue cosine = {cos(tem1_res[i], tem52_res[i]):.4f}")

# Whole-protein L2 distance (more sensitive than cosine for high-sim cases)
l2 = float(np.linalg.norm(tem1_pool.astype(np.float32) - tem52_pool.astype(np.float32)))
print(f"Mean-pooled L2 distance: {l2:.4f}")

# Discriminability test: TEM-1 vs random *non-beta-lactamase* protein
print("\n== Spread test: resistance vs random pairs ==")
prots = pd.concat([pd.read_parquet(f'work/raw/proteins_{sp}.parquet',
                                    columns=['aa_sequence_md5', 'aa_sequence', 'aa_length'])
                   for sp in ['ecoli', 'saureus', 'kpneumo']],
                   ignore_index=True).drop_duplicates(subset='aa_sequence_md5')

# Filter: have embedding
prots = prots[prots['aa_sequence_md5'].isin(embs.keys())].reset_index(drop=True)
print(f"Proteins with embeddings: {len(prots)}")

# Resistance group: AMRFinder-called BETA-LACTAM
betalac_md5 = set(md5_amr[md5_amr['Class'] == 'BETA-LACTAM']['Protein id'])
beta_in_embs = list(betalac_md5 & set(embs.keys()))
print(f"BETA-LACTAM md5s with embeddings: {len(beta_in_embs)}")

# Pick 10 resistance-vs-random pairs
np.random.seed(42)
non_resistance = list(set(embs.keys()) - set(md5_amr['Protein id']))
print(f"Non-resistance proteins available: {len(non_resistance)}")

cos_pairs = []
for trial in range(10):
    r = np.random.choice(beta_in_embs)
    o = np.random.choice(non_resistance)
    c = cos(embs[r], embs[o])
    cos_pairs.append({'resistance_md5': r, 'random_md5': o, 'cosine': c})
    print(f"  pair{trial+1}: cos = {c:.4f}")

cos_arr = np.array([p['cosine'] for p in cos_pairs])
print(f"\nResistance-vs-random cosines: mean={cos_arr.mean():.3f} std={cos_arr.std():.3f} min={cos_arr.min():.3f} max={cos_arr.max():.3f}")
print(f"All < 0.7? {(cos_arr < 0.7).all()}")
print(f"Fraction < 0.7: {(cos_arr < 0.7).mean()*100:.0f}%")

# Also: same-family within resistance (should be higher than random)
print("\n== Within-resistance spread (10 pairs of BETA-LACTAM proteins) ==")
within_pairs = []
for trial in range(10):
    a, b = np.random.choice(beta_in_embs, 2, replace=False)
    c = cos(embs[a], embs[b])
    within_pairs.append({'a_md5': a, 'b_md5': b, 'cosine': c})
    print(f"  pair{trial+1}: cos = {c:.4f}")
within_arr = np.array([p['cosine'] for p in within_pairs])
print(f"Within-BETA-LACTAM cosines: mean={within_arr.mean():.3f} std={within_arr.std():.3f}")

# Cross-family resistance: BETA-LACTAM vs QUINOLONE
quin_md5 = set(md5_amr[md5_amr['Class'] == 'QUINOLONE']['Protein id'])
quin_in_embs = list(quin_md5 & set(embs.keys()))
print(f"\n== Cross-family BETA-LACTAM vs QUINOLONE (10 pairs) ==")
cross_pairs = []
for trial in range(10):
    a = np.random.choice(beta_in_embs)
    b = np.random.choice(quin_in_embs)
    c = cos(embs[a], embs[b])
    cross_pairs.append({'a': a, 'b': b, 'cosine': c})
    print(f"  pair{trial+1}: cos = {c:.4f}")
cross_arr = np.array([p['cosine'] for p in cross_pairs])
print(f"Cross-family cosines: mean={cross_arr.mean():.3f} std={cross_arr.std():.3f}")

# ---- Resumability check
print("\n== Resumability check ==")
done = set(open(EMB_DIR / 'done_md5s.txt').read().split())
print(f"done_md5s.txt: {len(done)} unique md5s")
print(f"protein_embeddings.npz keys: {len(md5_keys)}")
print(f"Intersection: {len(done & set(md5_keys))}")

# Per-protein dim check
sample_md5 = md5_keys[0]
print(f"\nSample embedding shape: {embs[sample_md5].shape}, dtype: {embs[sample_md5].dtype}")

# Residue index
ridx = pd.read_parquet(EMB_DIR / 'residue_embeddings_index.parquet')
print(f"residue_embeddings_index: {len(ridx)} rows")
print(f"Residue files on disk: {len(list((EMB_DIR / 'residue_embeddings').glob('*.npy')))}")
sample_residue = list((EMB_DIR / 'residue_embeddings').glob('*.npy'))[0]
arr = np.load(sample_residue)
print(f"Sample residue file shape: {arr.shape}, dtype: {arr.dtype}")

# ---- Save validation summary
summary = {
    'tem1_tem52_mean_pooled_cos': float(cos(tem1_pool, tem52_pool)),
    'tem1_tem52_per_residue_cos_at_subs': {f"{i+1}_{a1}_{a52}": float(cos(tem1_res[i], tem52_res[i])) for (i, a1, a52) in subs},
    'tem1_tem52_l2': l2,
    'resistance_vs_random_cosines': cos_pairs,
    'resistance_vs_random_mean': float(cos_arr.mean()),
    'resistance_vs_random_all_lt_0_7': bool((cos_arr < 0.7).all()),
    'within_betalac_mean': float(within_arr.mean()),
    'cross_betalac_quinolone_mean': float(cross_arr.mean()),
    'n_embedded': len(md5_keys),
    'n_resistance_md5': len(resistance_md5),
    'n_resistance_covered': len(covered),
    'coverage_pct': len(covered) / max(len(resistance_md5), 1) * 100,
    'residue_n': len(ridx),
    'embedding_dim': int(embs[sample_md5].shape[0]),
    'dtype': str(embs[sample_md5].dtype),
}
with open(EMB_DIR / '_gate4_meta.json', 'w') as f:
    json.dump(summary, f, indent=2, default=float)
print(f"\nSaved validation summary -> {EMB_DIR / '_gate4_meta.json'}")
print(json.dumps(summary, indent=2, default=float))
