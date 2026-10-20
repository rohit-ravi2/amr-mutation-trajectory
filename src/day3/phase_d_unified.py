"""Phase D Day-3 — unified canonical-AMR-rich pipeline.

Steps:
  1. Build features with Day-2 learned projector for short-protein residues
     AND long-protein residues (already 128-dim) merged.
  2. Apply NEW canonical-AMR-rich split: train = lab transitions excluding clinical;
     test = clinical+CARD+curated trajectories.
  3. Retrain LightGBM LambdaRank on train. Evaluate on test.
  4. Retrain transformer on the SAME train transitions. Evaluate on test.
  5. Build the reranker: for each test transition take LightGBM top-10 candidates,
     extract transformer log-probs at the mutation tokens, train a LightGBM
     reranker (LambdaRank objective) on these features.

Outputs:
  work/models/day3/single_step_lgbm_d3.pkl
  work/models/day3/transformer_d3.pt
  work/models/day3/reranker_d3.pkl
  work/results/day3/phase_d_results.json
"""
from __future__ import annotations
import os, sys, json, pickle, re, time, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ------ paths
SSD = Path('work')
SRC = SSD / 'src'
RESULTS = SSD / 'results' / 'day3'
MODELS  = SSD / 'models' / 'day3'
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
SPLITS = SSD / 'processed' / 'splits'
EMB = SSD / 'embeddings'
LOG_PATH = SSD / 'logs' / 'day3' / 'phase_d.log'
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger('phaseD')
LOG.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)sZ %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
sh = logging.StreamHandler(); sh.setFormatter(fmt); LOG.addHandler(sh)
fh = logging.FileHandler(LOG_PATH, mode='w'); fh.setFormatter(fmt); LOG.addHandler(fh)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG.info(f'device={DEV}')

# ------ load split
with open(SPLITS / 'canonical_amr_split.pkl','rb') as f:
    split = pickle.load(f)
train_tids = split['train_transition_ids']
test_tids = split['test_transition_ids']
LOG.info(f'train n={len(train_tids)}, test n={len(test_tids)}, test_canonical={split["test_canonical_n"]}')

# ------ Day-2 learned residue projector
class ResidueAEv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1280, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 128),
        )
    def forward(self, x): return self.encoder(x)

def load_encoder():
    sd = torch.load(SSD / 'models' / 'residue_projector.pt', map_location='cpu', weights_only=False)
    e = ResidueAEv2Encoder()
    e.encoder.load_state_dict(sd['encoder_state_dict'])
    e.eval()
    return e

encoder = load_encoder().to(DEV)
LOG.info('Loaded Day-2 learned encoder (1280->128)')

# ------ Monkey-patch gate6_baseline
sys.path.insert(0, str(SRC))
import gate6_baseline as g6

class LearnedResidueProjector:
    """Drop-in for sklearn PCA — .transform(X) of (n,1280) -> (n,128)."""
    def __init__(self, enc, device):
        self.encoder = enc.to(device)
        self.device = device
    def transform(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            z = self.encoder(t).cpu().numpy()
        return z

learned = LearnedResidueProjector(encoder, DEV)

# Patch FeatureBuilder.get_residue_pca to also handle long-protein embeddings
LONG_DIR = EMB / 'residue_embeddings_long'

_original_get_residue = g6.FeatureBuilder.get_residue_pca

def patched_get_residue_pca(self, md5, pos1):
    """Hybrid: short residues in residue_dir (1280-dim, project), long in residue_embeddings_long/ (already 128-dim)."""
    if md5 is None or pos1 is None:
        return None
    seqlen = self._md5_seqlen.get(md5)
    if seqlen is None:
        return None
    if pos1 < 1 or pos1 > seqlen:
        return None
    if md5 not in self._residue_cache:
        # Try short dir first
        fp_short = self.residue_dir / f'{md5}.npy'
        fp_long  = LONG_DIR / f'{md5}.npy'
        if fp_short.exists():
            arr = np.load(fp_short).astype(np.float32, copy=False)
            self._residue_cache[md5] = ('short', arr)
        elif fp_long.exists():
            arr = np.load(fp_long).astype(np.float32, copy=False)  # already 128-dim
            self._residue_cache[md5] = ('long', arr)
        else:
            self._residue_cache[md5] = ('none', None)
    kind, arr = self._residue_cache[md5]
    if arr is None:
        return None
    idx = pos1 - 1
    if idx >= arr.shape[0]:
        return None
    v = arr[idx]
    if kind == 'short':
        v_red = self.residue_pca_model.transform(v.reshape(1, -1))[0]
        return v_red.astype(np.float32)
    else:  # 'long' — already 128-dim
        return v.astype(np.float32)

g6.FeatureBuilder.get_residue_pca = patched_get_residue_pca

# Patch FeatureBuilder to force n_residpca=128
_orig_init = g6.FeatureBuilder.__init__
def patched_init(self, protein_emb_pca, residue_emb_index, residue_dir,
                 protein_npz, dms_idx, gene_to_md5,
                 protein_pca_model, residue_pca_model,
                 protein_emb_dim_full=1280, n_proteinpca=64, n_residpca=32):
    _orig_init(self, protein_emb_pca, residue_emb_index, residue_dir,
               protein_npz, dms_idx, gene_to_md5,
               protein_pca_model, residue_pca_model,
               protein_emb_dim_full=protein_emb_dim_full,
               n_proteinpca=n_proteinpca, n_residpca=128)
g6.FeatureBuilder.__init__ = patched_init

# Patch fit_residue_pca to return our learned projector
def fit_residue_pca_stub(*args, **kwargs):
    LOG.info('[patched] returning learned 1280->128 projector')
    return learned
g6.fit_residue_pca = fit_residue_pca_stub

# ============================================================
# Replicate gate6 main pipeline up to feature build, then swap in our split
# ============================================================
from collections import Counter, defaultdict

def main():
    t0 = time.time()
    LOG.info('Loading data')
    trans = pd.read_parquet(SSD / 'processed' / 'transitions.parquet')
    traj  = pd.read_parquet(SSD / 'processed' / 'trajectories.parquet')

    # Hook organism from trajectory
    org_map = dict(zip(traj['trajectory_id'], traj['organism']))
    trans['organism'] = trans['trajectory_id'].map(org_map).fillna('Escherichia coli')

    # Apply NEW canonical-AMR-rich split
    trans['split'] = 'train'
    trans.loc[trans['transition_id'].astype(str).isin(test_tids), 'split'] = 'test_canonical'
    LOG.info('split sizes:\n%s', trans['split'].value_counts().to_string())

    # Restrict to recognized organisms / drug classes for now
    # vocabulary
    LOG.info('Building candidate vocabulary by drug_class...')
    # Vocab: all mutations seen in trans, per drug_class
    vocab_by_dc = defaultdict(set)
    for _, r in trans.iterrows():
        vocab_by_dc[r['drug_class']].add(r['mutation_added'])
    # Include cross-class small overlap so that test classes get reasonable vocab
    # Actually: also include each gene's known mutations from CARD if available
    # For now use class-level
    candidate_vocab = {k: sorted(v) for k,v in vocab_by_dc.items()}
    n_vocab = sum(len(v) for v in candidate_vocab.values())
    LOG.info(f'vocab size sum = {n_vocab}; per-class min={min(len(v) for v in candidate_vocab.values())}, max={max(len(v) for v in candidate_vocab.values())}')

    # ----- Load Gate-4/AMR md5/dms/embeddings using gate6 helpers
    # protein_npz
    protein_npz = np.load(EMB / 'protein_embeddings.npz')
    # residue_index — refreshed with long entries from Phase B
    residue_index = pd.read_parquet(EMB / 'residue_embeddings_index.parquet')
    LOG.info(f'residue_index: {len(residue_index)} rows; long (>512): {(residue_index["seq_len"]>512).sum()}')

    # md5_to_amr — gene -> md5 map via gate6 builder (list-of-dicts format)
    m2a = pd.read_parquet(EMB / '_md5_to_amr.parquet')
    gene_to_md5 = g6.build_gene_to_md5(m2a)
    # Add canonical gene aliases for gyrA, parC, rpoB, gyrB, parE, acrB
    canonical_kw = {
        'gyrA':['gyra','dna gyrase subunit a'],
        'gyrB':['gyrb','dna gyrase subunit b'],
        'parC':['parc','topoisomerase iv subunit a'],
        'parE':['pare','topoisomerase iv subunit b'],
        'rpoB':['rpob','rna polymerase beta'],
        'acrB':['acrb','multidrug efflux'],
        'marR':['marr'],
        'soxR':['soxr'],
    }
    for sp in ['ecoli','saureus','kpneumo']:
        try:
            p = pd.read_parquet(f'work/raw/proteins_{sp}.parquet',
                                columns=['gene','product','aa_sequence_md5'])
            for gene, kws in canonical_kw.items():
                if gene in gene_to_md5 and gene_to_md5[gene]:
                    continue
                for kw in kws:
                    hit = p[p['product'].astype(str).str.lower().str.contains(kw, regex=False, na=False)]
                    if len(hit):
                        md5 = hit['aa_sequence_md5'].value_counts().idxmax()
                        gene_to_md5.setdefault(gene, []).append({'md5': md5, 'species': sp, 'class': '', 'sym': gene})
                        break
        except Exception as e:
            LOG.warning(f'gene-md5 lookup err {sp}: {e}')
    LOG.info(f'gene_to_md5 entries: {len(gene_to_md5)}')

    # dms
    dms = pd.read_parquet(SSD / 'processed' / 'dms_fitness.parquet')
    LOG.info(f'dms: {len(dms)} rows')
    dms_idx = g6.build_dms_lookup(dms)

    # Protein PCA — use existing fit on protein_npz (64 dim)
    LOG.info('Fitting protein PCA (64 dim)')
    pca_p, md5_to_pc = g6.fit_protein_pca(protein_npz, n_components=64)

    # Build FeatureBuilder
    feat_builder = g6.FeatureBuilder(
        protein_emb_pca=md5_to_pc,
        residue_emb_index=residue_index,
        residue_dir=EMB / 'residue_embeddings',
        protein_npz=protein_npz,
        dms_idx=dms_idx,
        gene_to_md5=gene_to_md5,
        protein_pca_model=pca_p,
        residue_pca_model=learned,
        n_proteinpca=64,
        n_residpca=128,
    )

    # ----- mut_freq from train-split only
    train_only = trans[trans['split']=='train']
    mut_freq_in_train = Counter(train_only['mutation_added'].tolist())
    mut_freq_per_class = {}
    for dc, sub in train_only.groupby('drug_class'):
        mut_freq_per_class[dc] = Counter(sub['mutation_added'].tolist())
    LOG.info(f'mut_freq_in_train uniq muts: {len(mut_freq_in_train)}')

    # ----- Precompute candidate features
    LOG.info('Precomputing candidate features...')
    cand_cache = g6.precompute_candidate_features(
        candidate_vocab, feat_builder, dms_idx, gene_to_md5,
        mut_freq_in_train=mut_freq_in_train,
        mut_freq_per_class=mut_freq_per_class,
    )

    # ----- Build features (limit train negatives to 50)
    LOG.info('Building features...')
    X_all, Y_all, G_all, META_all, feat_names = g6.build_features(
        trans, candidate_vocab, feat_builder, dms_idx, gene_to_md5,
        max_neg_per_pos=50, split_col='split', cand_cache=cand_cache,
    )
    META_all = META_all.merge(trans[['transition_id','split','source_paper','lab_or_clinical','organism','is_hypermutator_trajectory']],
                              on='transition_id', how='left')
    LOG.info(f'X={X_all.shape}, Y_pos={int(Y_all.sum())}')

    # Sort by transition_id for groups
    order = np.argsort(META_all['transition_id'].values, kind='stable')
    X_s = X_all[order]
    Y_s = Y_all[order]
    META_s = META_all.iloc[order].reset_index(drop=True)

    train_mask = META_s['split'].values == 'train'
    test_mask = META_s['split'].values == 'test_canonical'
    LOG.info(f'train rows={int(train_mask.sum())}, test rows={int(test_mask.sum())}')

    def group_sizes(meta):
        return meta.groupby('transition_id', sort=False).size().values

    # Train/val split inside train
    train_meta = META_s[train_mask].reset_index(drop=True)
    train_tids_list = train_meta['transition_id'].unique()
    rng = np.random.default_rng(20260514)
    val_tids = set(rng.choice(train_tids_list, size=max(1, int(0.10*len(train_tids_list))), replace=False).tolist())
    val_mask_in_train = train_meta['transition_id'].isin(val_tids).values
    fit_mask = ~val_mask_in_train

    X_tr = X_s[train_mask]; Y_tr = Y_s[train_mask]
    X_fit = X_tr[fit_mask]; Y_fit = Y_tr[fit_mask]
    fit_meta = train_meta[fit_mask].reset_index(drop=True)
    fit_groups = group_sizes(fit_meta)
    X_val = X_tr[val_mask_in_train]; Y_val = Y_tr[val_mask_in_train]
    val_meta = train_meta[val_mask_in_train].reset_index(drop=True)
    val_groups = group_sizes(val_meta)
    LOG.info(f'fit transitions: {len(fit_groups)}, val: {len(val_groups)}')

    import lightgbm as lgb
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_at': [1,3,10],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'lambda_l2': 0.1,
        'verbose': -1,
        'max_depth': -1,
    }
    LOG.info(f'lgbm params: {params}')
    ds_tr  = lgb.Dataset(X_fit, label=Y_fit, group=fit_groups, feature_name=feat_names, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=Y_val, group=val_groups, feature_name=feat_names, free_raw_data=False)
    t_fit = time.time()
    lgbm = lgb.train(
        params, ds_tr, num_boost_round=500,
        valid_sets=[ds_tr, ds_val], valid_names=['train','val'],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True),
                   lgb.log_evaluation(period=25)],
    )
    LOG.info(f'lgbm best_iter={lgbm.best_iteration} fit_time={time.time()-t_fit:.1f}s')

    # Save lgbm
    with open(MODELS / 'single_step_lgbm_d3.pkl','wb') as f:
        pickle.dump({'model': lgbm, 'feature_names': feat_names, 'params': params}, f)
    LOG.info(f'saved lgbm -> {MODELS/"single_step_lgbm_d3.pkl"}')

    # ----- Evaluate lgbm on test
    X_test = X_s[test_mask]
    test_meta = META_s[test_mask].reset_index(drop=True)
    scores_lgbm = lgbm.predict(X_test)
    test_meta = test_meta.copy()
    test_meta['lgbm_score'] = scores_lgbm

    # Top-k accuracy helper
    def topk_acc(meta_subset, score_col, ks=(1,3,5,10)):
        acc = {k: 0 for k in ks}
        n = 0
        for tid, sub in meta_subset.groupby('transition_id', sort=False):
            if sub['is_positive'].sum() == 0:
                continue
            sub2 = sub.sort_values(score_col, ascending=False)
            ranks = np.where(sub2['is_positive'].values == 1)[0]
            if len(ranks) == 0:
                continue
            r = ranks[0]
            for k in ks:
                if r < k:
                    acc[k] += 1
            n += 1
        return {f'top{k}': acc[k]/max(n,1) for k in ks}, n

    overall_lgbm, n_overall = topk_acc(test_meta, 'lgbm_score')
    canon_meta = test_meta[test_meta['canonical_amr_flag']==1]
    canon_lgbm, n_canon = topk_acc(canon_meta, 'lgbm_score')
    LOG.info(f'LGBM overall: {overall_lgbm} (n={n_overall})')
    LOG.info(f'LGBM canonical: {canon_lgbm} (n={n_canon})')

    # Per drug class breakdown
    per_class_lgbm = {}
    for dc, sub in test_meta.groupby('drug_class'):
        acc, n_ = topk_acc(sub, 'lgbm_score')
        per_class_lgbm[dc] = {**acc, 'n': n_}

    # =====================================================
    # ----- Train a transformer on the SAME train set
    # =====================================================
    LOG.info('Building transformer training samples on the new train split...')
    # Reuse Day-2 transformer code structure but with new split
    sys.path.insert(0, str(SSD / 'src' / 'day2'))
    import phaseD_transformer_v2 as ptv2

    # Override assign_split to use our canonical-AMR-rich split (per-transition)
    def assign_split_canonical(transitions):
        # Return Series indexed by trajectory_id -> split
        by_traj = transitions.drop_duplicates('trajectory_id').set_index('trajectory_id')
        splits = pd.Series('train', index=by_traj.index)
        # If ANY transition in a trajectory is in test_tids, the WHOLE trajectory is test
        traj_to_split = {}
        for tid in transitions['trajectory_id'].unique():
            tids = set(transitions[transitions['trajectory_id']==tid]['transition_id'].astype(str))
            if tids & test_tids:
                traj_to_split[tid] = 'test_canonical'
            else:
                traj_to_split[tid] = 'train'
        for tid in by_traj.index:
            splits[tid] = traj_to_split.get(tid, 'train')
        return splits

    ptv2.assign_split = assign_split_canonical

    # Build samples
    vocab_tf = ptv2.build_vocab(trans, min_count=1)
    LOG.info(f'transformer vocab size: {len(vocab_tf)}')
    tok2id = {t:i for i,t in enumerate(vocab_tf)}

    # Build samples
    samples = ptv2.build_transition_samples(trans, traj, vocab_tf)
    LOG.info(f'transformer samples (one per transition): {len(samples)}')

    # Assign split label to each sample
    traj_split = assign_split_canonical(trans)
    for s in samples:
        s['split'] = traj_split.get(s['trajectory_id'], 'train')

    train_samples = [s for s in samples if s['split']=='train']
    test_samples  = [s for s in samples if s['split']=='test_canonical']
    LOG.info(f'transformer train={len(train_samples)} test={len(test_samples)}')

    # Build dms_lookup dict (matching Day-2 transformer signature)
    dms_lookup = {}
    for _, r in dms.iterrows():
        key = (r['gene_family'], r['position'], r['wildtype_aa'], r['mutant_aa'])
        dms_lookup[key] = float(r.get('fitness_zscore', 0.0))
    LOG.info(f'dms_lookup keys: {len(dms_lookup)}')

    # gene_to_md5 for transformer (flat dict: gene -> md5 string)
    gene_to_md5_tf = {}
    for sp in ['ecoli','saureus','kpneumo']:
        p = pd.read_parquet(f'work/raw/proteins_{sp}.parquet',
                            columns=['gene','aa_sequence_md5','aa_length'])
        p = p[p['gene'].notna() & p['aa_sequence_md5'].notna()]
        p = p.sort_values('aa_length')
        for _, r in p.iterrows():
            g = r['gene']; m = r['aa_sequence_md5']
            if g and m:
                gene_to_md5_tf.setdefault(str(g), str(m))
    # Extend with canonical from gene_to_md5 list-of-dicts
    for k, vlist in gene_to_md5.items():
        if vlist and isinstance(vlist, list):
            gene_to_md5_tf.setdefault(k, vlist[0]['md5'])
    LOG.info(f'gene_to_md5 (transformer flat): {len(gene_to_md5_tf)}')

    LOG.info('Precomputing transformer token features...')
    tok_feats = ptv2.precompute_token_features(
        vocab_tf, dms_lookup, gene_to_md5_tf, EMB / 'residue_embeddings',
        encoder, DEV, protein_npz=protein_npz,
    )
    LOG.info(f'tok_feats shape: {tok_feats.shape}')
    tok_feats_t = torch.tensor(tok_feats, dtype=torch.float32, device=DEV)

    # Build model
    model_tf = ptv2.TrajectoryTransformerV2(
        vocab_size=len(vocab_tf), d_model=128, n_layers=4, n_heads=4, ffn_dim=512,
        dropout=0.1, max_len=64, n_drug_classes=len(ptv2.DC_TO_IDX),
        n_organisms=len(ptv2.ORG_TO_IDX),
        token_feat_dim=tok_feats.shape[1], use_token_features=True,
    ).to(DEV)
    LOG.info(f'transformer params: {sum(p.numel() for p in model_tf.parameters())/1e6:.2f}M')

    # Train
    n_epochs = 50  # shorter than Day-2 (we have more train data and less time)
    pad_id = tok2id['<PAD>']
    rng_tf = np.random.default_rng(0)
    # Inner val: 10% of train samples
    train_tids_all = list({s['trajectory_id'] for s in train_samples})
    rng_tf.shuffle(train_tids_all)
    val_t = set(train_tids_all[: max(1, int(0.10*len(train_tids_all)))])
    train_only = [s for s in train_samples if s['trajectory_id'] not in val_t]
    val_only = [s for s in train_samples if s['trajectory_id'] in val_t]
    LOG.info(f'transformer fit n={len(train_only)} val n={len(val_only)}')

    opt = torch.optim.AdamW(model_tf.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    # Cross-entropy with class weights 5x for canonical mutations
    canon_mask = np.zeros(len(vocab_tf), dtype=np.float32)
    for tok in vocab_tf:
        if ptv2.is_canonical(tok):
            canon_mask[tok2id[tok]] = 1.0
    LOG.info(f'canonical tokens in vocab: {canon_mask.sum()}')

    BATCH = 64
    best_val_top3 = -1.0
    best_state = None
    for ep in range(n_epochs):
        model_tf.train()
        rng_tf.shuffle(train_only)
        losses = []
        for bs in range(0, len(train_only), BATCH):
            batch = train_only[bs:bs+BATCH]
            d = ptv2.collate(batch, pad_id=pad_id, max_len=32)
            in_ids = d['in_ids'].to(DEV)
            tgt = d['tgt'].to(DEV)
            drug = d['drug'].to(DEV)
            org = d['org'].to(DEV)
            step = d['step'].to(DEV)
            canon = d['canon'].to(DEV)
            logits = model_tf(in_ids, drug, org, step, token_feats=tok_feats_t)
            # weighted CE
            w = 1.0 + 4.0 * canon
            ce = F.cross_entropy(logits, tgt, reduction='none')
            loss = (ce * w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_tf.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        # Evaluate val
        model_tf.eval()
        with torch.no_grad():
            val_correct3 = 0; val_n = 0
            for bs in range(0, len(val_only), BATCH):
                batch = val_only[bs:bs+BATCH]
                if not batch: continue
                d = ptv2.collate(batch, pad_id=pad_id, max_len=32)
                in_ids = d['in_ids'].to(DEV)
                tgt = d['tgt'].to(DEV)
                drug = d['drug'].to(DEV); org = d['org'].to(DEV); step = d['step'].to(DEV)
                logits = model_tf(in_ids, drug, org, step, token_feats=tok_feats_t)
                # top3
                top3 = logits.topk(3, dim=-1).indices  # [B,3]
                hit = (top3 == tgt.unsqueeze(-1)).any(dim=-1).float()
                val_correct3 += int(hit.sum().item())
                val_n += hit.numel()
            val_top3 = val_correct3 / max(val_n,1)
        LOG.info(f'epoch {ep+1}/{n_epochs} train_loss={np.mean(losses):.4f} val_top3={val_top3:.4f}')
        if val_top3 > best_val_top3:
            best_val_top3 = val_top3
            best_state = {k: v.detach().cpu().clone() for k,v in model_tf.state_dict().items()}

    if best_state is not None:
        model_tf.load_state_dict(best_state)
    # save
    torch.save({
        'state_dict': model_tf.state_dict(),
        'vocab': vocab_tf,
        'best_val_top3': best_val_top3,
        'config': {'d_model':128,'n_layers':4,'n_heads':4,'ffn_dim':512,'dropout':0.1,'max_len':64},
    }, MODELS / 'transformer_d3.pt')
    LOG.info(f'transformer saved. best_val_top3={best_val_top3:.4f}')

    # ----- Evaluate transformer on test_canonical
    model_tf.eval()
    test_top1=0; test_top3=0; test_n=0
    canon_top1=0; canon_top3=0; canon_n=0
    # Also accumulate per-test sample log-probs over ALL vocab for reranker use
    transformer_logits_per_sample = {}  # (trajectory_id, step_number) -> [vocab_size] log-probs

    with torch.no_grad():
        for bs in range(0, len(test_samples), BATCH):
            batch = test_samples[bs:bs+BATCH]
            if not batch: continue
            d = ptv2.collate(batch, pad_id=pad_id, max_len=32)
            in_ids = d['in_ids'].to(DEV)
            tgt = d['tgt'].to(DEV)
            drug = d['drug'].to(DEV); org = d['org'].to(DEV); step = d['step'].to(DEV)
            canon = d['canon'].to(DEV)
            logits = model_tf(in_ids, drug, org, step, token_feats=tok_feats_t)
            log_probs = F.log_softmax(logits, dim=-1)
            top1 = logits.topk(1, dim=-1).indices.squeeze(-1)
            top3 = logits.topk(3, dim=-1).indices
            hit1 = (top1 == tgt).float()
            hit3 = (top3 == tgt.unsqueeze(-1)).any(dim=-1).float()
            test_top1 += int(hit1.sum().item()); test_top3 += int(hit3.sum().item())
            test_n += int(hit1.numel())
            # canonical-subset
            cm = canon.bool()
            canon_top1 += int(hit1[cm].sum().item())
            canon_top3 += int(hit3[cm].sum().item())
            canon_n += int(cm.sum().item())
            # store log-probs per sample for reranker
            for i, s in enumerate(batch):
                key = (s['trajectory_id'], s['step_number'])
                transformer_logits_per_sample[key] = log_probs[i].detach().cpu().numpy().astype(np.float32)

    tf_overall = {'top1': test_top1/max(test_n,1), 'top3': test_top3/max(test_n,1), 'n': test_n}
    tf_canonical = {'top1': canon_top1/max(canon_n,1), 'top3': canon_top3/max(canon_n,1), 'n': canon_n}
    LOG.info(f'TRANSFORMER overall: {tf_overall}')
    LOG.info(f'TRANSFORMER canonical: {tf_canonical}')

    # =====================================================
    # ----- Phase D Reranker: LightGBM top-10 + transformer log-prob
    # =====================================================
    LOG.info('Building reranker features (top-10 candidates per transition)')
    # For each test transition: rank candidates by lgbm score, take top-10
    # For each candidate: get transformer log-prob from per-sample dist
    test_meta_renamed = test_meta  # has lgbm_score
    # Build per-transition ordered list of (candidate, lgbm_score, lgbm_rank, is_positive, etc.)
    rerank_rows = []
    # Need to know which (traj, step) the transition corresponds to
    trans_lkp = trans.set_index('transition_id')[['trajectory_id','step_number','drug_class','mutation_added','mutation_in_canonical_amr_gene']]
    for tid, sub in test_meta_renamed.groupby('transition_id', sort=False):
        # Need transformer log-probs for (trajectory_id, step_number)
        meta_row = trans_lkp.loc[tid]
        traj_id = meta_row['trajectory_id']; step_n = int(meta_row['step_number'])
        tf_lp = transformer_logits_per_sample.get((traj_id, step_n))
        # If no transformer log-prob (sample skipped because canonical-not-in-vocab):
        # use zeros (informative but degenerate). We mark with has_tf=0.
        has_tf = 1 if tf_lp is not None else 0

        sub = sub.sort_values('lgbm_score', ascending=False).reset_index(drop=True)
        sub['lgbm_rank'] = sub.index + 1
        top10 = sub.head(10).copy()
        for i, cr in top10.iterrows():
            cand = cr['candidate']
            tid_token = tok2id.get(cand, tok2id['<UNK>'])
            tf_lp_val = float(tf_lp[tid_token]) if tf_lp is not None else 0.0
            rerank_rows.append({
                'transition_id': tid,
                'trajectory_id': traj_id,
                'step_number': step_n,
                'candidate': cand,
                'is_positive': int(cr['is_positive']),
                'lgbm_score': float(cr['lgbm_score']),
                'lgbm_rank': int(cr['lgbm_rank']),
                'tf_logprob': tf_lp_val,
                'has_tf': has_tf,
                'canonical_amr_flag': int(cr['canonical_amr_flag']),
                'drug_class': cr['drug_class'],
                'mut_freq_in_train': float(mut_freq_in_train.get(cand, 0)),
                'mut_freq_per_class': float(mut_freq_per_class.get(cr['drug_class'], {}).get(cand, 0)),
            })

    rerank_df = pd.DataFrame(rerank_rows)
    LOG.info(f'rerank rows (10/transition * n_tests): {len(rerank_df)}')
    # transformer rank within the same 10 candidates
    rerank_df['tf_rank_in_top10'] = rerank_df.groupby('transition_id')['tf_logprob'].rank(ascending=False, method='first').astype(int)

    # Need to also build held-out train-style data to train the reranker.
    # Strategy: also generate top-10 candidates for a held-out subset of training transitions
    # so the reranker has supervised training data. Use 10% of val transitions.
    LOG.info('Generating rerank training pool from internal val transitions...')
    # Predict on val_meta (already in our META_s)
    val_X = X_tr[val_mask_in_train]
    scores_val = lgbm.predict(val_X)
    val_meta_renamed = val_meta.copy(); val_meta_renamed['lgbm_score'] = scores_val
    # Also need transformer log-probs for VAL trajectories (we currently only computed for test).
    # Run transformer on val_only samples to get log-probs
    transformer_logits_val = {}
    with torch.no_grad():
        for bs in range(0, len(val_only), BATCH):
            batch = val_only[bs:bs+BATCH]
            if not batch: continue
            d = ptv2.collate(batch, pad_id=pad_id, max_len=32)
            in_ids = d['in_ids'].to(DEV)
            drug = d['drug'].to(DEV); org = d['org'].to(DEV); step = d['step'].to(DEV)
            logits = model_tf(in_ids, drug, org, step, token_feats=tok_feats_t)
            log_probs = F.log_softmax(logits, dim=-1)
            for i, s in enumerate(batch):
                key = (s['trajectory_id'], s['step_number'])
                transformer_logits_val[key] = log_probs[i].detach().cpu().numpy().astype(np.float32)

    # Build rerank train rows from val_meta
    rerank_train_rows = []
    for tid, sub in val_meta_renamed.groupby('transition_id', sort=False):
        if tid not in trans_lkp.index:
            continue
        meta_row = trans_lkp.loc[tid]
        traj_id = meta_row['trajectory_id']; step_n = int(meta_row['step_number'])
        tf_lp = transformer_logits_val.get((traj_id, step_n))
        has_tf = 1 if tf_lp is not None else 0
        sub = sub.sort_values('lgbm_score', ascending=False).reset_index(drop=True)
        sub['lgbm_rank'] = sub.index + 1
        top10 = sub.head(10).copy()
        for i, cr in top10.iterrows():
            cand = cr['candidate']
            tid_token = tok2id.get(cand, tok2id['<UNK>'])
            tf_lp_val = float(tf_lp[tid_token]) if tf_lp is not None else 0.0
            rerank_train_rows.append({
                'transition_id': tid,
                'trajectory_id': traj_id,
                'step_number': step_n,
                'candidate': cand,
                'is_positive': int(cr['is_positive']),
                'lgbm_score': float(cr['lgbm_score']),
                'lgbm_rank': int(cr['lgbm_rank']),
                'tf_logprob': tf_lp_val,
                'has_tf': has_tf,
                'canonical_amr_flag': int(cr['canonical_amr_flag']),
                'drug_class': cr['drug_class'],
                'mut_freq_in_train': float(mut_freq_in_train.get(cand, 0)),
                'mut_freq_per_class': float(mut_freq_per_class.get(cr['drug_class'], {}).get(cand, 0)),
            })
    rerank_train_df = pd.DataFrame(rerank_train_rows)
    rerank_train_df['tf_rank_in_top10'] = rerank_train_df.groupby('transition_id')['tf_logprob'].rank(ascending=False, method='first').astype(int)
    LOG.info(f'rerank train rows: {len(rerank_train_df)}')

    # Reranker feature columns
    drug_classes_list = sorted(rerank_train_df['drug_class'].astype(str).unique().tolist() + rerank_df['drug_class'].astype(str).unique().tolist())
    drug_classes_list = sorted(set(drug_classes_list))
    DC2I = {dc:i for i,dc in enumerate(drug_classes_list)}

    def feat_block(df):
        # numeric features
        num = df[['lgbm_score','lgbm_rank','tf_logprob','tf_rank_in_top10','has_tf','canonical_amr_flag','mut_freq_in_train','mut_freq_per_class']].values.astype(np.float32)
        # drug class oh
        dc_oh = np.zeros((len(df), len(drug_classes_list)), dtype=np.float32)
        for i, dc in enumerate(df['drug_class'].astype(str).values):
            if dc in DC2I:
                dc_oh[i, DC2I[dc]] = 1.0
        return np.hstack([num, dc_oh])

    X_rtr = feat_block(rerank_train_df)
    Y_rtr = rerank_train_df['is_positive'].values.astype(np.int32)
    g_rtr = rerank_train_df.groupby('transition_id', sort=False).size().values

    X_rte = feat_block(rerank_df)
    Y_rte = rerank_df['is_positive'].values.astype(np.int32)
    g_rte = rerank_df.groupby('transition_id', sort=False).size().values

    feat_names_r = ['lgbm_score','lgbm_rank','tf_logprob','tf_rank_in_top10','has_tf','canonical_amr_flag',
                    'mut_freq_in_train','mut_freq_per_class'] + [f'dc_{dc}' for dc in drug_classes_list]

    LOG.info(f'Reranker train shape: {X_rtr.shape}, test shape: {X_rte.shape}')
    LOG.info(f'Reranker n_train_groups: {len(g_rtr)}, n_test_groups: {len(g_rte)}')

    # Train reranker (LambdaRank over top-10)
    rer_params = {
        'objective':'lambdarank','metric':'ndcg','ndcg_at':[1,3,10],
        'learning_rate':0.05,'num_leaves':31,'min_data_in_leaf':10,
        'feature_fraction':0.9,'bagging_fraction':0.9,'bagging_freq':5,
        'lambda_l2':0.5,'verbose':-1,
    }
    ds_rtr = lgb.Dataset(X_rtr, label=Y_rtr, group=g_rtr, feature_name=feat_names_r, free_raw_data=False)
    # 10% val from rtr
    n_rtr_tr = len(rerank_train_df['transition_id'].unique())
    rng_r = np.random.default_rng(42)
    rtr_tids = list(rerank_train_df['transition_id'].unique())
    rng_r.shuffle(rtr_tids)
    rtr_val = set(rtr_tids[: max(1,int(0.10*len(rtr_tids)))])
    val_idx = rerank_train_df['transition_id'].isin(rtr_val).values
    fit_idx = ~val_idx
    X_rfit, Y_rfit = X_rtr[fit_idx], Y_rtr[fit_idx]
    fit_g_r = rerank_train_df[fit_idx].groupby('transition_id', sort=False).size().values
    X_rval, Y_rval = X_rtr[val_idx], Y_rtr[val_idx]
    val_g_r = rerank_train_df[val_idx].groupby('transition_id', sort=False).size().values
    ds_rfit = lgb.Dataset(X_rfit, label=Y_rfit, group=fit_g_r, feature_name=feat_names_r, free_raw_data=False)
    ds_rval = lgb.Dataset(X_rval, label=Y_rval, group=val_g_r, feature_name=feat_names_r, free_raw_data=False)
    LOG.info(f'reranker fit rows: {X_rfit.shape}, val rows: {X_rval.shape}')
    rer = lgb.train(rer_params, ds_rfit, num_boost_round=200,
                    valid_sets=[ds_rfit, ds_rval], valid_names=['train','val'],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False),
                               lgb.log_evaluation(period=25)])
    LOG.info(f'reranker best_iter={rer.best_iteration}')
    with open(MODELS / 'reranker_d3.pkl','wb') as f:
        pickle.dump({'model': rer, 'feature_names': feat_names_r,
                     'drug_classes_list': drug_classes_list, 'params': rer_params}, f)

    # ----- Evaluate reranker on test
    rer_scores = rer.predict(X_rte)
    rerank_df['rerank_score'] = rer_scores

    overall_rer, n_overall_r = topk_acc(rerank_df, 'rerank_score')
    canon_rer_meta = rerank_df[rerank_df['canonical_amr_flag']==1]
    canon_rer, n_canon_r = topk_acc(canon_rer_meta, 'rerank_score')
    LOG.info(f'RERANKER overall: {overall_rer} (n={n_overall_r})')
    LOG.info(f'RERANKER canonical: {canon_rer} (n={n_canon_r})')

    # Also compute transformer-only on the same top-10 set
    overall_tfonly, _ = topk_acc(rerank_df, 'tf_logprob')
    canon_tfonly, _ = topk_acc(canon_rer_meta, 'tf_logprob')
    # And LGBM-only on same top-10 set (for matched comparison)
    overall_lgbm_only_top10, _ = topk_acc(rerank_df, 'lgbm_score')
    canon_lgbm_only_top10, _ = topk_acc(canon_rer_meta, 'lgbm_score')

    # Per drug class for rerank
    per_class_rer = {}
    for dc, sub in rerank_df.groupby('drug_class'):
        acc, n_ = topk_acc(sub, 'rerank_score')
        per_class_rer[dc] = {**acc, 'n': n_}

    # Save reranker results
    rerank_df.to_parquet(RESULTS / 'phase_d_reranker_test_predictions.parquet', index=False)

    final = {
        'split_info': {
            'n_train_transitions': len(train_tids),
            'n_test_transitions': len(test_tids),
            'n_test_canonical': split['test_canonical_n'],
        },
        'lgbm_only_full_vocab': {
            'overall': {**overall_lgbm, 'n': n_overall},
            'canonical': {**canon_lgbm, 'n': n_canon},
            'per_drug_class': per_class_lgbm,
        },
        'transformer_only_full_vocab': {
            'overall': tf_overall,
            'canonical': tf_canonical,
        },
        'lgbm_only_top10_subset': {
            'overall': {**overall_lgbm_only_top10},
            'canonical': {**canon_lgbm_only_top10},
        },
        'transformer_only_top10_subset': {
            'overall': {**overall_tfonly},
            'canonical': {**canon_tfonly},
        },
        'reranker_top10': {
            'overall': {**overall_rer, 'n': n_overall_r},
            'canonical': {**canon_rer, 'n': n_canon_r},
            'per_drug_class': per_class_rer,
        },
        'reranker_features': feat_names_r,
        'wall_clock_min': (time.time()-t0)/60.0,
    }

    with open(RESULTS / 'phase_d_results.json','w') as f:
        json.dump(final, f, indent=2, default=str)

    LOG.info(json.dumps({k:v for k,v in final.items() if k != 'lgbm_only_full_vocab'}, indent=2, default=str))

    # Gate log line
    gl = SSD/'logs'/'gates.log'
    ts = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    target_met = bool(overall_rer.get('top3',0) >= 0.70 and canon_rer.get('top3',0) >= 0.40)
    with gl.open('a') as f:
        f.write(f'{ts} PHASE_D_DAY3: [reranker_overall_top3={overall_rer.get("top3",0):.4f}] '
                f'[reranker_canonical_top3={canon_rer.get("top3",0):.4f}] '
                f'[lgbm_overall_top3={overall_lgbm.get("top3",0):.4f}] '
                f'[lgbm_canonical_top3={canon_lgbm.get("top3",0):.4f}] '
                f'[transformer_overall_top3={tf_overall.get("top3",0):.4f}] '
                f'[transformer_canonical_top3={tf_canonical.get("top3",0):.4f}] '
                f'[target_>=0.70_and_>=0.40={target_met}]\n')

    print(json.dumps({
        'reranker_overall_top3': float(overall_rer.get('top3',0)),
        'reranker_canonical_top3': float(canon_rer.get('top3',0)),
        'lgbm_overall_top3': float(overall_lgbm.get('top3',0)),
        'lgbm_canonical_top3': float(canon_lgbm.get('top3',0)),
        'transformer_overall_top3': float(tf_overall.get('top3',0)),
        'transformer_canonical_top3': float(tf_canonical.get('top3',0)),
        'target_met': target_met,
    }, indent=2))
    return final

if __name__ == '__main__':
    main()
