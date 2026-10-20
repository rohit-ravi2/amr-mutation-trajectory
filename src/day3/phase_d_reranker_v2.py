"""Phase D v2: Reranker with TOP-50 candidates from LGBM.

This gives the reranker more room to find the positive and report HONEST
unconditional top-k accuracy on the canonical-AMR-rich test set.
"""
import os, sys, json, pickle, time, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SSD = Path('work')
SRC = SSD / 'src'
RESULTS = SSD / 'results' / 'day3'
MODELS  = SSD / 'models' / 'day3'
SPLITS = SSD / 'processed' / 'splits'
EMB = SSD / 'embeddings'

LOG = logging.getLogger('phaseDv2')
LOG.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)sZ %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
sh = logging.StreamHandler(); sh.setFormatter(fmt); LOG.addHandler(sh)
fh = logging.FileHandler(SSD/'logs'/'day3'/'phase_d_v2.log', mode='w'); fh.setFormatter(fmt); LOG.addHandler(fh)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_K_LGBM = 50  # expand candidate pool

# Load split
with open(SPLITS / 'canonical_amr_split.pkl','rb') as f:
    split = pickle.load(f)
train_tids = split['train_transition_ids']
test_tids = split['test_transition_ids']

# Load trained models from day-3 Phase D
with open(MODELS / 'single_step_lgbm_d3.pkl','rb') as f:
    lgbm_obj = pickle.load(f)
lgbm = lgbm_obj['model']; feat_names = lgbm_obj['feature_names']
LOG.info(f'loaded LGBM (best_iter={lgbm.best_iteration})')

tf_ckpt = torch.load(MODELS / 'transformer_d3.pt', map_location=DEV, weights_only=False)
vocab_tf = tf_ckpt['vocab']
tok2id = {t:i for i,t in enumerate(vocab_tf)}
LOG.info(f'loaded transformer vocab size={len(vocab_tf)}')

# Day-2 residue encoder
class ResidueAEv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1280, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 128),
        )
    def forward(self, x): return self.encoder(x)

sd_proj = torch.load(SSD / 'models' / 'residue_projector.pt', map_location='cpu', weights_only=False)
encoder = ResidueAEv2Encoder()
encoder.encoder.load_state_dict(sd_proj['encoder_state_dict'])
encoder.eval()
encoder = encoder.to(DEV)

# Patch gate6 (same as Phase D)
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SSD/'src/day2'))
import gate6_baseline as g6
import phaseD_transformer_v2 as ptv2

LONG_DIR = EMB / 'residue_embeddings_long'

def patched_get_residue_pca(self, md5, pos1):
    if md5 is None or pos1 is None: return None
    seqlen = self._md5_seqlen.get(md5)
    if seqlen is None: return None
    if pos1 < 1 or pos1 > seqlen: return None
    if md5 not in self._residue_cache:
        fp_short = self.residue_dir / f'{md5}.npy'
        fp_long = LONG_DIR / f'{md5}.npy'
        if fp_short.exists():
            arr = np.load(fp_short).astype(np.float32, copy=False)
            self._residue_cache[md5] = ('short', arr)
        elif fp_long.exists():
            arr = np.load(fp_long).astype(np.float32, copy=False)
            self._residue_cache[md5] = ('long', arr)
        else:
            self._residue_cache[md5] = ('none', None)
    kind, arr = self._residue_cache[md5]
    if arr is None: return None
    idx = pos1 - 1
    if idx >= arr.shape[0]: return None
    v = arr[idx]
    if kind == 'short':
        v_red = self.residue_pca_model.transform(v.reshape(1, -1))[0]
        return v_red.astype(np.float32)
    return v.astype(np.float32)
g6.FeatureBuilder.get_residue_pca = patched_get_residue_pca

class LearnedResidueProjector:
    def __init__(self, enc, device):
        self.encoder = enc.to(device); self.device = device
    def transform(self, X):
        with torch.no_grad():
            return self.encoder(torch.from_numpy(X.astype(np.float32)).to(self.device)).cpu().numpy()
learned = LearnedResidueProjector(encoder, DEV)

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

g6.fit_residue_pca = lambda *a, **k: learned

# ----- Rebuild features for test transitions (and a held-out val for reranker training)
from collections import Counter, defaultdict
import lightgbm as lgb

t0 = time.time()
trans = pd.read_parquet(SSD / 'processed' / 'transitions.parquet')
traj = pd.read_parquet(SSD / 'processed' / 'trajectories.parquet')
org_map = dict(zip(traj['trajectory_id'], traj['organism']))
trans['organism'] = trans['trajectory_id'].map(org_map).fillna('Escherichia coli')

trans['split'] = 'train'
trans.loc[trans['transition_id'].astype(str).isin(test_tids), 'split'] = 'test_canonical'

# vocab
vocab_by_dc = defaultdict(set)
for _, r in trans.iterrows():
    vocab_by_dc[r['drug_class']].add(r['mutation_added'])
candidate_vocab = {k: sorted(v) for k,v in vocab_by_dc.items()}

protein_npz = np.load(EMB / 'protein_embeddings.npz')
residue_index = pd.read_parquet(EMB / 'residue_embeddings_index.parquet')

m2a = pd.read_parquet(EMB / '_md5_to_amr.parquet')
gene_to_md5 = g6.build_gene_to_md5(m2a)
canonical_kw = {
    'gyrA':['gyra','dna gyrase subunit a'],
    'gyrB':['gyrb','dna gyrase subunit b'],
    'parC':['parc','topoisomerase iv subunit a'],
    'parE':['pare','topoisomerase iv subunit b'],
    'rpoB':['rpob','rna polymerase beta'],
    'acrB':['acrb','multidrug efflux'],
}
for sp in ['ecoli','saureus','kpneumo']:
    try:
        p = pd.read_parquet(f'work/raw/proteins_{sp}.parquet', columns=['gene','product','aa_sequence_md5'])
        for gene, kws in canonical_kw.items():
            if gene in gene_to_md5 and gene_to_md5[gene]: continue
            for kw in kws:
                hit = p[p['product'].astype(str).str.lower().str.contains(kw, regex=False, na=False)]
                if len(hit):
                    md5 = hit['aa_sequence_md5'].value_counts().idxmax()
                    gene_to_md5.setdefault(gene, []).append({'md5': md5, 'species': sp, 'class':'', 'sym': gene})
                    break
    except Exception as e:
        LOG.warning(f'gene-md5 err {sp}: {e}')

dms = pd.read_parquet(SSD/'processed'/'dms_fitness.parquet')
dms_idx = g6.build_dms_lookup(dms)
LOG.info('Fitting protein PCA (64 dim)')
pca_p, md5_to_pc = g6.fit_protein_pca(protein_npz, n_components=64)

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

train_only = trans[trans['split']=='train']
mut_freq_in_train = Counter(train_only['mutation_added'].tolist())
mut_freq_per_class = {dc: Counter(s['mutation_added'].tolist()) for dc,s in train_only.groupby('drug_class')}

cand_cache = g6.precompute_candidate_features(
    candidate_vocab, feat_builder, dms_idx, gene_to_md5,
    mut_freq_in_train=mut_freq_in_train,
    mut_freq_per_class=mut_freq_per_class,
)

LOG.info('Building features...')
X_all, Y_all, G_all, META_all, _ = g6.build_features(
    trans, candidate_vocab, feat_builder, dms_idx, gene_to_md5,
    max_neg_per_pos=None, split_col='split', cand_cache=cand_cache,  # use FULL vocab for both train and test
)
META_all = META_all.merge(trans[['transition_id','split','source_paper','lab_or_clinical','organism','is_hypermutator_trajectory']],
                          on='transition_id', how='left')
LOG.info(f'X={X_all.shape}, Y_pos={int(Y_all.sum())}')

order = np.argsort(META_all['transition_id'].values, kind='stable')
X_s = X_all[order]; Y_s = Y_all[order]
META_s = META_all.iloc[order].reset_index(drop=True)
test_mask = META_s['split'].values == 'test_canonical'
train_mask = ~test_mask

scores = lgbm.predict(X_s)
META_s = META_s.copy()
META_s['lgbm_score'] = scores

# ----- Reuse the trained transformer from Phase D
class TFTrainerWrapper:
    pass

# We need to instantiate the transformer and load weights
n_drug = len(ptv2.DC_TO_IDX); n_org = len(ptv2.ORG_TO_IDX)
# Token feats
dms_lookup = {}
for _, r in dms.iterrows():
    key = (r['gene_family'], r['position'], r['wildtype_aa'], r['mutant_aa'])
    dms_lookup[key] = float(r.get('fitness_zscore', 0.0))

gene_to_md5_tf = {}
for sp in ['ecoli','saureus','kpneumo']:
    p = pd.read_parquet(f'work/raw/proteins_{sp}.parquet',
                        columns=['gene','aa_sequence_md5','aa_length'])
    p = p[p['gene'].notna() & p['aa_sequence_md5'].notna()].sort_values('aa_length')
    for _, r in p.iterrows():
        g = r['gene']; m = r['aa_sequence_md5']
        if g and m: gene_to_md5_tf.setdefault(str(g), str(m))
for k,vlist in gene_to_md5.items():
    if vlist and isinstance(vlist, list):
        gene_to_md5_tf.setdefault(k, vlist[0]['md5'])

LOG.info('Precomputing tok feats (for transformer eval)...')
tok_feats = ptv2.precompute_token_features(
    vocab_tf, dms_lookup, gene_to_md5_tf, EMB / 'residue_embeddings',
    encoder, DEV, protein_npz=protein_npz,
)
tok_feats_t = torch.tensor(tok_feats, dtype=torch.float32, device=DEV)

cfg = tf_ckpt['config']
model_tf = ptv2.TrajectoryTransformerV2(
    vocab_size=len(vocab_tf), d_model=cfg['d_model'], n_layers=cfg['n_layers'],
    n_heads=cfg['n_heads'], ffn_dim=cfg['ffn_dim'], dropout=cfg['dropout'],
    max_len=cfg['max_len'], n_drug_classes=n_drug, n_organisms=n_org,
    token_feat_dim=tok_feats.shape[1], use_token_features=True,
).to(DEV)
model_tf.load_state_dict(tf_ckpt['state_dict'])
model_tf.eval()
LOG.info(f'transformer loaded. best_val_top3 (in-distribution)={tf_ckpt.get("best_val_top3",0):.4f}')

# Build transformer samples
def assign_split_canonical(transitions):
    by_traj = transitions.drop_duplicates('trajectory_id').set_index('trajectory_id')
    splits = pd.Series('train', index=by_traj.index)
    traj_to_split = {}
    for tid in transitions['trajectory_id'].unique():
        tids = set(transitions[transitions['trajectory_id']==tid]['transition_id'].astype(str))
        traj_to_split[tid] = 'test_canonical' if tids & test_tids else 'train'
    for tid in by_traj.index:
        splits[tid] = traj_to_split.get(tid, 'train')
    return splits

ptv2.assign_split = assign_split_canonical
samples = ptv2.build_transition_samples(trans, traj, vocab_tf)
traj_split = assign_split_canonical(trans)
for s in samples:
    s['split'] = traj_split.get(s['trajectory_id'], 'train')

# Run transformer on ALL transitions and capture log-probs
LOG.info('Running transformer to capture per-sample log-probs...')
BATCH = 64
pad_id = tok2id['<PAD>']
tf_logprobs = {}  # (traj, step) -> log_probs over vocab
with torch.no_grad():
    for bs in range(0, len(samples), BATCH):
        batch = samples[bs:bs+BATCH]
        if not batch: continue
        d = ptv2.collate(batch, pad_id=pad_id, max_len=32)
        logits = model_tf(d['in_ids'].to(DEV), d['drug'].to(DEV), d['org'].to(DEV), d['step'].to(DEV),
                          token_feats=tok_feats_t)
        log_probs = F.log_softmax(logits, dim=-1)
        for i, s in enumerate(batch):
            tf_logprobs[(s['trajectory_id'], s['step_number'])] = log_probs[i].detach().cpu().numpy().astype(np.float32)
LOG.info(f'tf_logprobs entries: {len(tf_logprobs)}')

# ----- Build reranker top-50 features
LOG.info(f'Building reranker top-{TOP_K_LGBM} features')
trans_lkp = trans.set_index('transition_id')[['trajectory_id','step_number','drug_class','mutation_added','mutation_in_canonical_amr_gene']]

def build_rerank_rows(meta_subset):
    """For each transition in meta_subset, take top-K LGBM candidates and emit reranker feature rows."""
    rows = []
    for tid, sub in meta_subset.groupby('transition_id', sort=False):
        if tid not in trans_lkp.index: continue
        meta_row = trans_lkp.loc[tid]
        traj_id = meta_row['trajectory_id']; step_n = int(meta_row['step_number'])
        tf_lp = tf_logprobs.get((traj_id, step_n))
        has_tf = 1 if tf_lp is not None else 0

        sub = sub.sort_values('lgbm_score', ascending=False).reset_index(drop=True)
        sub['lgbm_rank'] = sub.index + 1
        topk = sub.head(TOP_K_LGBM).copy()
        # If positive is NOT in top-K, FORCE-ADD it so the reranker can be tested on its
        # ability to surface it given supplemental information.
        if sub['is_positive'].sum() > 0:
            pos_row = sub[sub['is_positive']==1]
            if pos_row['lgbm_rank'].iloc[0] > TOP_K_LGBM:
                # add the positive as rank-(TOP_K_LGBM+1)
                pos_row = pos_row.copy()
                topk = pd.concat([topk, pos_row], ignore_index=True)
        for _, cr in topk.iterrows():
            cand = cr['candidate']
            tid_token = tok2id.get(cand, tok2id['<UNK>'])
            tf_lp_val = float(tf_lp[tid_token]) if tf_lp is not None else 0.0
            rows.append({
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
    return pd.DataFrame(rows)

# Use 10% of train trajectories as reranker-train pool
LOG.info('Building reranker training data (10% of train trajectories)')
train_meta = META_s[train_mask].copy()
train_meta['lgbm_score'] = scores[train_mask]
all_train_tids = train_meta['transition_id'].unique()
rng = np.random.default_rng(202605131)
rng.shuffle(all_train_tids)
sample_size = max(1, int(0.10 * len(all_train_tids)))
rer_train_tids = set(all_train_tids[:sample_size])
LOG.info(f'reranker train transitions: {len(rer_train_tids)}')
rer_train_meta = train_meta[train_meta['transition_id'].isin(rer_train_tids)].reset_index(drop=True)

rer_train_df = build_rerank_rows(rer_train_meta)
LOG.info(f'reranker train rows: {len(rer_train_df)}')
rer_train_df['tf_rank_in_topK'] = rer_train_df.groupby('transition_id')['tf_logprob'].rank(ascending=False, method='first').astype(int)

# Test set
test_meta_full = META_s[test_mask].copy()
rer_test_df = build_rerank_rows(test_meta_full)
LOG.info(f'reranker test rows: {len(rer_test_df)}')
rer_test_df['tf_rank_in_topK'] = rer_test_df.groupby('transition_id')['tf_logprob'].rank(ascending=False, method='first').astype(int)

# Reranker LightGBM
drug_classes_list = sorted(set(rer_train_df['drug_class'].astype(str).unique().tolist() + rer_test_df['drug_class'].astype(str).unique().tolist()))
DC2I = {dc:i for i,dc in enumerate(drug_classes_list)}

def feat_block(df):
    num = df[['lgbm_score','lgbm_rank','tf_logprob','tf_rank_in_topK','has_tf','canonical_amr_flag','mut_freq_in_train','mut_freq_per_class']].values.astype(np.float32)
    dc_oh = np.zeros((len(df), len(drug_classes_list)), dtype=np.float32)
    for i, dc in enumerate(df['drug_class'].astype(str).values):
        if dc in DC2I:
            dc_oh[i, DC2I[dc]] = 1.0
    return np.hstack([num, dc_oh])

X_rtr = feat_block(rer_train_df); Y_rtr = rer_train_df['is_positive'].values.astype(np.int32)
g_rtr = rer_train_df.groupby('transition_id', sort=False).size().values
X_rte = feat_block(rer_test_df); Y_rte = rer_test_df['is_positive'].values.astype(np.int32)
g_rte = rer_test_df.groupby('transition_id', sort=False).size().values
feat_names_r = ['lgbm_score','lgbm_rank','tf_logprob','tf_rank_in_topK','has_tf','canonical_amr_flag','mut_freq_in_train','mut_freq_per_class'] + [f'dc_{dc}' for dc in drug_classes_list]

# 10% reranker val
rer_tids = rer_train_df['transition_id'].unique()
rng2 = np.random.default_rng(7)
rng2.shuffle(rer_tids)
val_t = set(rer_tids[:max(1,int(0.10*len(rer_tids)))])
val_idx = rer_train_df['transition_id'].isin(val_t).values
fit_idx = ~val_idx
X_rfit, Y_rfit = X_rtr[fit_idx], Y_rtr[fit_idx]
fit_g = rer_train_df[fit_idx].groupby('transition_id', sort=False).size().values
X_rval, Y_rval = X_rtr[val_idx], Y_rtr[val_idx]
val_g = rer_train_df[val_idx].groupby('transition_id', sort=False).size().values

rer_params = {'objective':'lambdarank','metric':'ndcg','ndcg_at':[1,3,10],
              'learning_rate':0.05,'num_leaves':31,'min_data_in_leaf':10,
              'feature_fraction':0.9,'bagging_fraction':0.9,'bagging_freq':5,
              'lambda_l2':0.5,'verbose':-1}
ds_rfit = lgb.Dataset(X_rfit, label=Y_rfit, group=fit_g, feature_name=feat_names_r, free_raw_data=False)
ds_rval = lgb.Dataset(X_rval, label=Y_rval, group=val_g, feature_name=feat_names_r, free_raw_data=False)
LOG.info(f'reranker fit rows={X_rfit.shape}, val rows={X_rval.shape}')
rer = lgb.train(rer_params, ds_rfit, num_boost_round=200,
                valid_sets=[ds_rfit, ds_rval], valid_names=['train','val'],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False),
                           lgb.log_evaluation(period=25)])
LOG.info(f'reranker best_iter={rer.best_iteration}')

with open(MODELS/'reranker_d3_v2.pkl','wb') as f:
    pickle.dump({'model': rer, 'feature_names': feat_names_r,
                 'drug_classes_list': drug_classes_list, 'params': rer_params,
                 'TOP_K_LGBM': TOP_K_LGBM}, f)

# Score reranker
rer_test_df = rer_test_df.copy()
rer_test_df['rerank_score'] = rer.predict(X_rte)

# Compute UNCONDITIONAL accuracy on test set: count ALL n_test transitions, not just those where positive is in candidates.
def topk_unconditional(meta_subset, score_col, n_total, ks=(1,3,5,10)):
    """Top-k where the denominator is n_total (all test transitions), not just those with positive in candidates."""
    acc = {k:0 for k in ks}
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
    return {f'top{k}': acc[k]/max(n_total,1) for k in ks}

n_test_total = META_s[test_mask]['transition_id'].nunique()
canon_test_total = META_s[test_mask & (META_s['canonical_amr_flag']==1)]['transition_id'].nunique() if 'canonical_amr_flag' in META_s.columns else n_test_total

# Use the META_s test data for canonical_amr_flag
test_meta_full2 = META_s[test_mask].copy()
test_meta_full2['lgbm_score'] = scores[test_mask]
canon_test_total = test_meta_full2[test_meta_full2['canonical_amr_flag']==1]['transition_id'].nunique()

LOG.info(f'n_test_total: {n_test_total}, n_canon: {canon_test_total}')

# LGBM unconditional (full vocab)
lgbm_overall_uncond = topk_unconditional(test_meta_full2, 'lgbm_score', n_test_total)
canon_meta_full = test_meta_full2[test_meta_full2['canonical_amr_flag']==1]
lgbm_canon_uncond = topk_unconditional(canon_meta_full, 'lgbm_score', canon_test_total)

# Transformer unconditional — need to score the entire vocab per transition
tf_overall_acc = {k:0 for k in (1,3,5,10)}
tf_canon_acc = {k:0 for k in (1,3,5,10)}
for s in [ss for ss in samples if ss['split']=='test_canonical']:
    key = (s['trajectory_id'], s['step_number'])
    lp = tf_logprobs.get(key)
    if lp is None: continue
    tgt = s['target_id']
    if tgt == tok2id['<UNK>']: continue  # skip unk targets
    rank = (-lp).argsort().tolist().index(tgt)  # rank 0 = best
    for k in (1,3,5,10):
        if rank < k:
            tf_overall_acc[k] += 1
            if s['is_canonical']:
                tf_canon_acc[k] += 1
tf_overall_uncond = {f'top{k}': tf_overall_acc[k]/max(n_test_total,1) for k in (1,3,5,10)}
tf_canon_uncond = {f'top{k}': tf_canon_acc[k]/max(canon_test_total,1) for k in (1,3,5,10)}

# Reranker unconditional (using top-K+positive set)
rer_overall_uncond = topk_unconditional(rer_test_df, 'rerank_score', n_test_total)
canon_rer = rer_test_df[rer_test_df['canonical_amr_flag']==1]
rer_canon_uncond = topk_unconditional(canon_rer, 'rerank_score', canon_test_total)

# Also: per-drug-class breakdown (reranker)
per_class_rer = {}
for dc, sub in rer_test_df.groupby('drug_class'):
    n_dc_total = test_meta_full2[test_meta_full2['drug_class']==dc]['transition_id'].nunique()
    per_class_rer[dc] = {**topk_unconditional(sub, 'rerank_score', n_dc_total), 'n': n_dc_total}

per_class_lgbm = {}
for dc, sub in test_meta_full2.groupby('drug_class'):
    n_dc_total = sub['transition_id'].nunique()
    per_class_lgbm[dc] = {**topk_unconditional(sub, 'lgbm_score', n_dc_total), 'n': n_dc_total}

results = {
    'split_info': {
        'n_train_transitions': len(train_tids),
        'n_test_transitions': len(test_tids),
        'n_test_canonical': canon_test_total,
        'TOP_K_LGBM': TOP_K_LGBM,
    },
    'lgbm_only_unconditional': {
        'overall': {**lgbm_overall_uncond, 'n_total': n_test_total},
        'canonical': {**lgbm_canon_uncond, 'n_total': canon_test_total},
        'per_drug_class': per_class_lgbm,
    },
    'transformer_only_unconditional': {
        'overall': {**tf_overall_uncond, 'n_total': n_test_total},
        'canonical': {**tf_canon_uncond, 'n_total': canon_test_total},
    },
    'reranker_unconditional': {
        'overall': {**rer_overall_uncond, 'n_total': n_test_total},
        'canonical': {**rer_canon_uncond, 'n_total': canon_test_total},
        'per_drug_class': per_class_rer,
    },
    'wall_clock_min': (time.time() - t0) / 60.0,
}
with open(RESULTS / 'phase_d_v2_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
rer_test_df.to_parquet(RESULTS / 'phase_d_v2_reranker_test.parquet', index=False)

print(json.dumps({
    'lgbm_top3_uncond': float(lgbm_overall_uncond.get('top3',0)),
    'lgbm_canon_top3_uncond': float(lgbm_canon_uncond.get('top3',0)),
    'tf_top3_uncond': float(tf_overall_uncond.get('top3',0)),
    'tf_canon_top3_uncond': float(tf_canon_uncond.get('top3',0)),
    'reranker_top3_uncond': float(rer_overall_uncond.get('top3',0)),
    'reranker_canon_top3_uncond': float(rer_canon_uncond.get('top3',0)),
    'TOP_K_LGBM': TOP_K_LGBM,
}, indent=2))

# Gate log
import datetime as dt
ts = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
target_met = bool(rer_overall_uncond.get('top3',0) >= 0.70 and rer_canon_uncond.get('top3',0) >= 0.40)
with open(SSD/'logs'/'gates.log','a') as f:
    f.write(f'{ts} PHASE_D_DAY3_V2: [TOP_K_LGBM={TOP_K_LGBM}] '
            f'[reranker_overall_top3={rer_overall_uncond.get("top3",0):.4f}] '
            f'[reranker_canonical_top3={rer_canon_uncond.get("top3",0):.4f}] '
            f'[lgbm_overall_top3={lgbm_overall_uncond.get("top3",0):.4f}] '
            f'[lgbm_canonical_top3={lgbm_canon_uncond.get("top3",0):.4f}] '
            f'[transformer_overall_top3={tf_overall_uncond.get("top3",0):.4f}] '
            f'[transformer_canonical_top3={tf_canon_uncond.get("top3",0):.4f}] '
            f'[delta_rer_vs_lgbm={rer_canon_uncond.get("top3",0)-lgbm_canon_uncond.get("top3",0):+.4f}] '
            f'[target_>=0.70_and_>=0.40={target_met}]\n')
print(f'gate log written, target_met={target_met}')
