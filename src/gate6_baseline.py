"""
Gate 6 — Single-step mutation prediction baseline.

LambdaRank (LightGBM) that ranks candidate next-mutations given
(current genotype, drug).  Held-out evaluations:
  primary  E. coli x ciprofloxacin/norfloxacin (full fluoroquinolone)
  withinALE  one ALEdb-124 sub-experiment (e1226) held out from FQ
  secondary S. aureus x oxacillin (anecdotal; tiny test set)
  secondary K. pneumoniae x meropenem (anecdotal; tiny test set)

Inputs:
  work/processed/transitions.parquet
  work/processed/trajectories.parquet
  work/processed/dms_fitness.parquet
  work/processed/resistance_genes.parquet
  work/embeddings/protein_embeddings.npz
  work/embeddings/residue_embeddings/<md5>.npy
  work/embeddings/residue_embeddings_index.parquet
  work/embeddings/_md5_to_amr.parquet

Outputs:
  work/models/single_step_lgbm.pkl
  work/models/esm2_pca_protein.pkl
  work/models/esm2_pca_residue.pkl
  work/results/single_step_eval.json
  work/results/single_step_features.csv (feature gains)
"""

import os, sys, re, json, pickle, time, logging, hashlib
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# --- paths -------------------------------------------------------------------
ROOT = Path("work")
PROC = ROOT / "processed"
EMB = ROOT / "embeddings"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("gate6")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "gate6.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

NP_RNG = np.random.default_rng(20260513)

# ----------------------------------------------------------------------------
# Mutation parsing
# ----------------------------------------------------------------------------
RE_AA = re.compile(r"^(.+)_([A-Z\*])(\d+)([A-Z\*])$")

def parse_mut(s):
    """Return (gene, kind, payload) for a mutation_added string.

    kind in {aa, coding, ig, pseudo, other}.
    payload is (wt, pos, mut) for aa, else None.
    """
    if s is None or not isinstance(s, str) or s == "" or s == "wildtype":
        return ("", "wildtype", None)
    for suf in ("_coding", "_ig", "_pseudo"):
        if s.endswith(suf):
            return (s[: -len(suf)], suf[1:], None)
    m = RE_AA.match(s)
    if m:
        return (m.group(1), "aa", (m.group(2), int(m.group(3)), m.group(4)))
    return (s, "other", None)


# ----------------------------------------------------------------------------
# Build candidate vocabulary per drug_class
# ----------------------------------------------------------------------------
def build_candidate_vocab(transitions, min_count_per_class=2,
                          canonical_amr_mutations=None):
    """For each drug_class, the candidate vocabulary is:
       all mutations occurring at least min_count_per_class times in that class
       PLUS all canonical-AMR-gene mutations seen in the dataset.
    """
    canonical = set(canonical_amr_mutations or [])
    vocab = {}
    for dc, sub in transitions.groupby("drug_class"):
        mc = sub["mutation_added"].value_counts()
        v = set(mc[mc >= min_count_per_class].index.tolist())
        # always include canonical AMR mutations observed in any class
        v |= canonical
        vocab[dc] = sorted(v)
    return vocab


# ----------------------------------------------------------------------------
# DMS lookup
# ----------------------------------------------------------------------------
def build_dms_lookup(dms):
    """Index DMS by (gene_family_lower, position, wt, mut) → (zscore, raw,
       source_normalization)."""
    g = dms.copy()
    g["gene_lower"] = g["gene_family"].astype(str).str.lower()
    # Use mean across datasets if multiple z-scores exist for the same key
    key = list(zip(g["gene_lower"], g["position"].astype(int),
                   g["wildtype_aa"].astype(str), g["mutant_aa"].astype(str)))
    g["__key"] = key
    agg = g.groupby("__key").agg(
        fit_z=("fitness_zscore", "mean"),
        fit_raw=("fitness_score", "mean"),
        source_norm=("source_normalization", "first"),
        n=("fitness_score", "size"),
    ).to_dict("index")
    return agg


# folA gene in trajectory data corresponds to DHFR DMS
GENE_TO_DMSFAMILY = {
    "folA": "dhfr",
    "blaTEM-1": "tem",
    "tem": "tem",
    "bla": "tem",
    "vim": "vim",
}
def dms_lookup(dms_idx, gene, pos, wt, mut):
    # Map trajectory gene to DMS gene_family
    fam = GENE_TO_DMSFAMILY.get(gene, gene.lower())
    rec = dms_idx.get((fam, int(pos), wt, mut))
    return rec  # may be None


# ----------------------------------------------------------------------------
# Resistance protein resolution
# ----------------------------------------------------------------------------
def build_gene_to_md5(md5_to_amr):
    """Map gene_symbol -> list of md5 candidates (per species)."""
    # Element symbol can be plain gene_symbol (oqxA), or already include
    # a known AMR mutation (parC_S80F). Build canonical gene name.
    g2m = defaultdict(list)
    for _, row in md5_to_amr.iterrows():
        sym = row["Element symbol"]
        md5 = row["Protein id"]
        sp = row["species"]
        cls = row["Class"]
        # parse mutation suffix
        m = RE_AA.match(sym)
        canonical = m.group(1) if m else sym
        # canonical may contain bla, gyrA, parC, ...
        g2m[canonical].append({"md5": md5, "species": sp, "class": cls, "sym": sym})
        # Also store a 2nd form without 'bla' prefix etc., as fallback
    return dict(g2m)


def resolve_protein_md5(gene, gene_to_md5):
    """Return a representative md5 for the protein (or None)."""
    keys = [gene, gene.upper(), gene.lower()]
    # E. coli specific: folA -> dfrA-like (NOT same as DHFR DMS). For
    # gyrA/parC AMR table only stores variant rows like "gyrA_S84L".
    aliases = {
        "folA": ["dfrB", "dfrA"],
        "gyrA": ["gyrA"], "gyrB": ["gyrB"],
        "parC": ["parC"], "parE": ["parE"],
        "rpoB": ["rpoB"],
        "marR": [], "acrR": [], "acrB": [],
    }
    for k in keys + aliases.get(gene, []):
        if k in gene_to_md5 and gene_to_md5[k]:
            # prefer plain symbol over annotated variant
            return gene_to_md5[k][0]["md5"]
    # search by prefix for gene families like parC_S80F
    matches = [v for k, v in gene_to_md5.items() if k.lower().startswith(gene.lower())]
    if matches:
        return matches[0][0]["md5"]
    return None


# ----------------------------------------------------------------------------
# Feature builder
# ----------------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self, protein_emb_pca, residue_emb_index, residue_dir,
                 protein_npz, dms_idx, gene_to_md5,
                 protein_pca_model, residue_pca_model,
                 protein_emb_dim_full=1280, n_proteinpca=64, n_residpca=32):
        self.protein_emb_pca = protein_emb_pca   # md5 -> 64-dim fp32
        self.residue_index = residue_emb_index   # df md5/seq_len
        self.residue_dir = residue_dir
        self.protein_npz = protein_npz           # NpzFile
        self.dms_idx = dms_idx
        self.gene_to_md5 = gene_to_md5
        self.protein_pca_model = protein_pca_model
        self.residue_pca_model = residue_pca_model
        self.n_pca_p = n_proteinpca
        self.n_pca_r = n_residpca
        # cache residue embeddings on demand (md5 -> fp32 arr)
        self._residue_cache = {}
        # pre-known seq_len lookup
        self._md5_seqlen = dict(zip(residue_emb_index["md5"], residue_emb_index["seq_len"]))

    def get_protein_pca(self, md5):
        v = self.protein_emb_pca.get(md5)
        return v  # may be None

    def get_residue_pca(self, md5, pos1):
        """Return PCA-reduced residue embedding at position pos1 (1-indexed)."""
        if md5 is None or pos1 is None:
            return None
        seqlen = self._md5_seqlen.get(md5)
        if seqlen is None:
            return None
        if pos1 < 1 or pos1 > seqlen:
            return None
        if md5 not in self._residue_cache:
            fp = self.residue_dir / f"{md5}.npy"
            if not fp.exists():
                self._residue_cache[md5] = None
                return None
            arr = np.load(fp).astype(np.float32, copy=False)
            self._residue_cache[md5] = arr
        arr = self._residue_cache[md5]
        if arr is None:
            return None
        # Residue arr could be shape [seqlen, 1280] or [seqlen+2, 1280] depending
        # on how the embedder stored it; assume [seqlen, 1280] (gate4 reported
        # 11382 files for proteins <=512 residues).
        idx = pos1 - 1
        if idx >= arr.shape[0]:
            return None
        v = arr[idx]  # (1280,)
        v_red = self.residue_pca_model.transform(v.reshape(1, -1))[0]
        return v_red.astype(np.float32)


# ----------------------------------------------------------------------------
# Pre-compute protein embedding PCA
# ----------------------------------------------------------------------------
def fit_protein_pca(protein_npz, n_components=64):
    keys = protein_npz.files
    LOG.info("Stacking %d protein embeddings for PCA (%d dim)...", len(keys), n_components)
    X = np.zeros((len(keys), 1280), dtype=np.float32)
    for i, k in enumerate(keys):
        X[i] = protein_npz[k].astype(np.float32, copy=False)
    pca = PCA(n_components=n_components, random_state=0)
    Xp = pca.fit_transform(X)
    md5_to_pc = {k: Xp[i] for i, k in enumerate(keys)}
    LOG.info("Protein PCA: explained variance = %.3f", pca.explained_variance_ratio_.sum())
    return pca, md5_to_pc


def fit_residue_pca(residue_index, residue_dir, n_components=32,
                    sample_max=200000, rng=NP_RNG):
    """Fit residue PCA on a random sample of residue vectors."""
    files = list(residue_index["md5"])
    # Subsample residue vectors across many proteins to fit PCA cheaply
    sample = []
    pool_size = 0
    target = sample_max
    rng.shuffle(files)
    for md5 in files:
        fp = residue_dir / f"{md5}.npy"
        if not fp.exists():
            continue
        a = np.load(fp).astype(np.float32, copy=False)
        # pick at most 30 random residues per protein
        k = min(30, a.shape[0])
        idx = rng.choice(a.shape[0], size=k, replace=False)
        sample.append(a[idx])
        pool_size += k
        if pool_size >= target:
            break
    X = np.concatenate(sample, axis=0)
    LOG.info("Residue PCA training set: %s", X.shape)
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X)
    LOG.info("Residue PCA: explained variance = %.3f", pca.explained_variance_ratio_.sum())
    return pca


# ----------------------------------------------------------------------------
# Build per-transition candidate sets and feature matrices
# ----------------------------------------------------------------------------
DRUG_CLASSES = [
    "fluoroquinolone", "diaminopyrimidine", "tetracycline",
    "cephalosporin", "penicillin", "aminoglycoside", "phenicol", "rifamycin",
]
DC_TO_IDX = {dc: i for i, dc in enumerate(DRUG_CLASSES)}

CANONICAL_AMR_GENES = {
    "gyrA", "gyrB", "parC", "parE", "folA", "rpoB",
    "marR", "acrR", "acrB", "acrA", "tolC", "ompC", "ompF",
    "soxR", "soxS", "marA", "rpoB", "rpoC", "uvrA", "envZ",
    "ftsI", "blaTEM-1", "blaSHV", "blaCTX-M", "blaOXA", "mecA",
    "tetA", "tetB", "tetM",
}

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA20)}


def precompute_candidate_features(candidate_vocab, feat_builder, dms_idx, gene_to_md5,
                                   mut_freq_in_train=None, mut_freq_per_class=None):
    """For each unique candidate mutation across all classes, pre-compute the
    candidate-only feature block (protein PCA, residue PCA, DMS, AA one-hots,
    candidate kind, gene info, canonical AMR indicator,
    plus mutation frequency in training data — overall and per drug class).

    Returns dict: mut_str -> dict with all candidate-only features
    """
    mut_freq_in_train = mut_freq_in_train or {}
    mut_freq_per_class = mut_freq_per_class or {}
    all_muts = set()
    for v in candidate_vocab.values():
        all_muts.update(v)
    LOG.info("Pre-computing candidate features for %d unique mutations...", len(all_muts))
    cache = {}
    t0 = time.time()
    for i, m in enumerate(all_muts):
        if i % 1000 == 0:
            LOG.info("  %d/%d (%.1fs)", i, len(all_muts), time.time() - t0)
        cg, ck, cpayl = parse_mut(m)
        cand_is_aa = (ck == "aa")
        cand_is_coding = (ck == "coding")
        cand_is_ig = (ck == "ig")
        cand_is_pseudo = (ck == "pseudo")
        cand_is_other = (ck == "other")
        cand_md5 = resolve_protein_md5(cg, gene_to_md5)
        protein_pca = (feat_builder.get_protein_pca(cand_md5)
                       if cand_md5 else None)
        has_protein_pca = 1
        if protein_pca is None:
            protein_pca = np.zeros(feat_builder.n_pca_p, dtype=np.float32)
            has_protein_pca = 0
        residue_pca = None
        if cand_is_aa and cand_md5 is not None:
            residue_pca = feat_builder.get_residue_pca(cand_md5, cpayl[1])
        has_residue_pca = 1
        if residue_pca is None:
            residue_pca = np.zeros(feat_builder.n_pca_r, dtype=np.float32)
            has_residue_pca = 0
        cand_dms_z = np.nan
        cand_dms_raw = np.nan
        cand_dms_norm = -1
        if cand_is_aa:
            rec = dms_lookup(dms_idx, cg, cpayl[1], cpayl[0], cpayl[2])
            if rec is not None:
                cand_dms_z = rec["fit_z"]
                cand_dms_raw = rec["fit_raw"]
                sn = rec.get("source_norm") or ""
                cand_dms_norm = hash(sn) % 16
        wt_oh = np.zeros(20, dtype=np.float32)
        mut_oh = np.zeros(20, dtype=np.float32)
        pos_norm = -1.0
        position = -1
        if cand_is_aa:
            w, p, mt = cpayl
            if w in AA_IDX: wt_oh[AA_IDX[w]] = 1.0
            if mt in AA_IDX: mut_oh[AA_IDX[mt]] = 1.0
            position = p
            if cand_md5 is not None:
                sl = feat_builder._md5_seqlen.get(cand_md5)
                if sl: pos_norm = min(p / sl, 1.0)
        cand_canon = int(cg in CANONICAL_AMR_GENES)
        freq_overall = float(mut_freq_in_train.get(m, 0))
        freq_per_class_vec = np.zeros(len(DRUG_CLASSES), dtype=np.float32)
        for dc, dcmap in mut_freq_per_class.items():
            if dc in DC_TO_IDX and m in dcmap:
                freq_per_class_vec[DC_TO_IDX[dc]] = float(dcmap[m])
        cache[m] = {
            "gene": cg,
            "kind": ck,
            "position": position,
            "is_aa": cand_is_aa,
            "kind_oh": np.array([cand_is_aa, cand_is_coding, cand_is_ig,
                                 cand_is_pseudo, cand_is_other],
                                dtype=np.float32),
            "protein_pca": protein_pca,
            "has_protein_pca": np.float32(has_protein_pca),
            "residue_pca": residue_pca,
            "has_residue_pca": np.float32(has_residue_pca),
            "dms_z": np.float32(cand_dms_z),
            "dms_raw": np.float32(cand_dms_raw),
            "dms_norm": np.float32(cand_dms_norm),
            "wt_oh": wt_oh,
            "mut_oh": mut_oh,
            "pos_norm": np.float32(pos_norm),
            "cand_canon": np.float32(cand_canon),
            "freq_overall": np.float32(freq_overall),
            "freq_per_class": freq_per_class_vec,
        }
    LOG.info("Pre-compute done in %.1fs", time.time() - t0)
    return cache


def build_features(transitions, candidate_vocab,
                   feat_builder, dms_idx, gene_to_md5,
                   gene_lengths=None,
                   max_neg_per_pos=None,
                   split_col=None,
                   rng=None,
                   cand_cache=None):
    """
    For each transition, build candidate (positive + negatives), compute features.

    If max_neg_per_pos is set, randomly subsample that many negatives per train
    transition. Test rows always use the full vocab so evaluation is realistic.

    cand_cache is the output of precompute_candidate_features().

    Returns X (n_rows, n_feat), y (n_rows,), groups (transition_id per row),
    plus metadata.
    """
    rng = rng or NP_RNG
    if cand_cache is None:
        cand_cache = precompute_candidate_features(candidate_vocab, feat_builder,
                                                    dms_idx, gene_to_md5)
    X_rows = []
    Y = []
    G = []
    META = []  # one dict per row
    feature_names = None

    # Pre-parse all mutations
    parsed_cache = {}
    def parsed(m):
        if m not in parsed_cache:
            parsed_cache[m] = parse_mut(m)
        return parsed_cache[m]

    # For each drug_class, the candidate vocab (mut_str list)
    n_t = len(transitions)
    t0 = time.time()
    for ti, row in enumerate(transitions.itertuples(index=False)):
        if ti % 500 == 0:
            LOG.info("build_features %d/%d (%.1fs elapsed)", ti, n_t, time.time() - t0)
        tid = row.transition_id
        traj_id = row.trajectory_id
        from_g = row.from_genotype
        true_mut = row.mutation_added
        drug = row.drug
        drug_class = row.drug_class
        step_num = row.step_number
        canon_flag = bool(row.mutation_in_canonical_amr_gene)

        # Decompose from_genotype
        if from_g is None or from_g == "wildtype" or from_g == "" :
            present = []
        else:
            present = from_g.split("+")
        present_set = set(present)
        n_bg = len(present)
        # parse each present mutation to (gene, kind, payload)
        present_parsed = [parsed(p) for p in present]
        present_aa_pos = defaultdict(list)  # gene -> list of int positions
        for g, k, payl in present_parsed:
            if k == "aa" and payl is not None:
                present_aa_pos[g].append(int(payl[1]))

        # Candidate set = drug-class vocabulary minus already-present mutations
        vocab = candidate_vocab.get(drug_class, [])
        cand = [c for c in vocab if c not in present_set]
        if true_mut not in cand and true_mut not in present_set:
            # force-add
            cand = cand + [true_mut]
            forced = True
        else:
            forced = False
        if true_mut in present_set:
            # actual mutation already present? shouldn't happen, skip
            continue

        # Subsample negatives for training rows only (to keep RAM manageable)
        # Identify whether this transition is in a train split
        is_train_row = True
        if split_col is not None:
            sv = getattr(row, split_col, "train")
            is_train_row = (sv == "train")
        if is_train_row and max_neg_per_pos is not None and len(cand) > max_neg_per_pos + 1:
            # keep positive + sample of negatives
            neg_pool = [c for c in cand if c != true_mut]
            if len(neg_pool) > max_neg_per_pos:
                neg_idx = rng.choice(len(neg_pool), size=max_neg_per_pos, replace=False)
                neg_pool = [neg_pool[i] for i in neg_idx]
            cand = neg_pool + [true_mut]

        # Compute features for each candidate
        # Pre-compute "current focal protein" PCA = mean of present-mutation
        # proteins' PCAs (proxy for genotype state)
        focal_pca = None
        if present_aa_pos:
            pcas = []
            for g_ in present_aa_pos.keys():
                md5_ = resolve_protein_md5(g_, gene_to_md5)
                if md5_ is not None:
                    v = feat_builder.get_protein_pca(md5_)
                    if v is not None:
                        pcas.append(v)
            if pcas:
                focal_pca = np.mean(pcas, axis=0)
        if focal_pca is None:
            focal_pca = np.zeros(feat_builder.n_pca_p, dtype=np.float32)

        # Sum-of-DMS-component fitness for from_genotype
        from_dms_sum = 0.0
        n_from_dms = 0
        for g, k, payl in present_parsed:
            if k == "aa" and payl is not None:
                rec = dms_lookup(dms_idx, g, payl[1], payl[0], payl[2])
                if rec is not None:
                    from_dms_sum += rec.get("fit_z", 0.0)
                    n_from_dms += 1

        # Pairwise epistasis sum: for each pair of present AA mutations,
        # if both in DMS, sum their z-scores' product / sum
        pair_epi_sum = 0.0
        n_pair_with_dms = 0
        present_aa_list = [(g, payl) for g, k, payl in present_parsed
                            if k == "aa" and payl is not None]
        for i_a in range(len(present_aa_list)):
            ga, pa = present_aa_list[i_a]
            ra = dms_lookup(dms_idx, ga, pa[1], pa[0], pa[2])
            if ra is None:
                continue
            for j_a in range(i_a + 1, len(present_aa_list)):
                gb, pb = present_aa_list[j_a]
                rb = dms_lookup(dms_idx, gb, pb[1], pb[0], pb[2])
                if rb is None:
                    continue
                pair_epi_sum += ra["fit_z"] + rb["fit_z"]
                n_pair_with_dms += 1

        # Drug-class one-hot (transition-level)
        dc_oh = np.zeros(len(DRUG_CLASSES), dtype=np.float32)
        if drug_class in DC_TO_IDX:
            dc_oh[DC_TO_IDX[drug_class]] = 1.0
        # transition-level scalars
        trans_scalars = np.array([from_dms_sum, n_from_dms,
                                  pair_epi_sum, n_pair_with_dms,
                                  step_num, n_bg], dtype=np.float32)

        # Ensure cand_cache has entries for the true mutation (force-add case)
        if true_mut not in cand_cache:
            # add it on the fly
            cg_, ck_, cpayl_ = parse_mut(true_mut)
            md5_ = resolve_protein_md5(cg_, gene_to_md5)
            ppca = feat_builder.get_protein_pca(md5_) if md5_ else None
            has_pp = 1
            if ppca is None:
                ppca = np.zeros(feat_builder.n_pca_p, dtype=np.float32)
                has_pp = 0
            rpca = None
            if ck_ == "aa" and md5_ is not None:
                rpca = feat_builder.get_residue_pca(md5_, cpayl_[1])
            has_rp = 1
            if rpca is None:
                rpca = np.zeros(feat_builder.n_pca_r, dtype=np.float32)
                has_rp = 0
            wt_oh_ = np.zeros(20, dtype=np.float32)
            mut_oh_ = np.zeros(20, dtype=np.float32)
            pos_norm_ = -1.0
            position_ = -1
            if ck_ == "aa":
                w_, p_, mt_ = cpayl_
                if w_ in AA_IDX: wt_oh_[AA_IDX[w_]] = 1.0
                if mt_ in AA_IDX: mut_oh_[AA_IDX[mt_]] = 1.0
                position_ = p_
                if md5_ is not None:
                    sl = feat_builder._md5_seqlen.get(md5_)
                    if sl: pos_norm_ = min(p_ / sl, 1.0)
            dms_z_ = np.nan
            dms_raw_ = np.nan
            dms_norm_ = -1
            if ck_ == "aa":
                rec_ = dms_lookup(dms_idx, cg_, cpayl_[1], cpayl_[0], cpayl_[2])
                if rec_ is not None:
                    dms_z_ = rec_["fit_z"]; dms_raw_ = rec_["fit_raw"]
                    sn_ = rec_.get("source_norm") or ""
                    dms_norm_ = hash(sn_) % 16
            cand_cache[true_mut] = {
                "gene": cg_, "kind": ck_,
                "position": position_,
                "is_aa": ck_ == "aa",
                "kind_oh": np.array([ck_=="aa", ck_=="coding", ck_=="ig",
                                     ck_=="pseudo", ck_=="other"], dtype=np.float32),
                "protein_pca": ppca,
                "has_protein_pca": np.float32(has_pp),
                "residue_pca": rpca,
                "has_residue_pca": np.float32(has_rp),
                "dms_z": np.float32(dms_z_),
                "dms_raw": np.float32(dms_raw_),
                "dms_norm": np.float32(dms_norm_),
                "wt_oh": wt_oh_,
                "mut_oh": mut_oh_,
                "pos_norm": np.float32(pos_norm_),
                "cand_canon": np.float32(int(cg_ in CANONICAL_AMR_GENES)),
                "freq_overall": np.float32(0),
                "freq_per_class": np.zeros(len(DRUG_CLASSES), dtype=np.float32),
            }

        for cand_mut in cand:
            cc = cand_cache[cand_mut]
            cg = cc["gene"]
            cpos = cc["position"]
            # Distance to nearest existing mutation in same gene
            dist_to_existing = -1.0
            if cc["is_aa"] and cg in present_aa_pos and cpos > 0:
                dist_to_existing = min(abs(cpos - x) for x in present_aa_pos[cg])

            # Build vector: ordered as feature_names will be
            vec = np.concatenate([
                focal_pca,                  # focal_p_*  (64)
                cc["protein_pca"],          # cand_p_*  (64)
                np.array([cc["has_protein_pca"]], dtype=np.float32),
                cc["residue_pca"],          # cand_r_*  (32)
                np.array([cc["has_residue_pca"]], dtype=np.float32),
                np.array([cc["dms_z"], cc["dms_raw"], cc["dms_norm"]],
                         dtype=np.float32),
                trans_scalars[:4],          # from_dms_sum, n_from_dms, pair_epi_sum, n_pair_with_dms
                dc_oh,                      # drug class
                trans_scalars[4:6],         # step_number, n_mut_background
                cc["wt_oh"],
                cc["mut_oh"],
                np.array([cc["pos_norm"], dist_to_existing], dtype=np.float32),
                cc["kind_oh"],
                np.array([cc["cand_canon"]], dtype=np.float32),
                np.array([cc["freq_overall"]], dtype=np.float32),
                cc["freq_per_class"],
            ])

            if feature_names is None:
                names = []
                names += [f"focal_p_{i}" for i in range(feat_builder.n_pca_p)]
                names += [f"cand_p_{i}" for i in range(feat_builder.n_pca_p)]
                names += ["has_protein_pca"]
                names += [f"cand_r_{i}" for i in range(feat_builder.n_pca_r)]
                names += ["has_residue_pca"]
                names += ["cand_dms_z", "cand_dms_raw", "cand_dms_normcat"]
                names += ["from_dms_sum", "n_from_dms",
                          "pair_epi_sum", "n_pair_with_dms"]
                names += [f"dc_{dc}" for dc in DRUG_CLASSES]
                names += ["step_number", "n_mut_background"]
                names += [f"wt_{a}" for a in AA20]
                names += [f"mut_{a}" for a in AA20]
                names += ["pos_norm", "dist_to_existing"]
                names += ["kind_aa", "kind_coding", "kind_ig",
                          "kind_pseudo", "kind_other"]
                names += ["cand_canonical_amr"]
                names += ["mut_freq_in_train"]
                names += [f"mut_freq_{dc}" for dc in DRUG_CLASSES]
                feature_names = names
                # sanity check dim
                assert len(feature_names) == vec.shape[0], \
                    f"name/vec mismatch: {len(feature_names)} vs {vec.shape[0]}"

            X_rows.append(vec)
            Y.append(1 if cand_mut == true_mut else 0)
            G.append(tid)
            META.append({
                "transition_id": tid,
                "trajectory_id": traj_id,
                "candidate": cand_mut,
                "is_positive": int(cand_mut == true_mut),
                "drug": drug,
                "drug_class": drug_class,
                "step_number": int(step_num),
                "n_bg": n_bg,
                "canonical_amr_flag": int(canon_flag),
                "forced_positive": int(forced and cand_mut == true_mut),
            })
    LOG.info("Total feature rows: %d", len(X_rows))
    X = np.vstack(X_rows).astype(np.float32)
    Y = np.array(Y, dtype=np.int32)
    G = np.array(G, dtype=object)
    META_df = pd.DataFrame(META)
    return X, Y, G, META_df, feature_names


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def topk_accuracy(meta_df, scores, ks=(1, 3, 10)):
    """meta_df has columns transition_id, is_positive. scores aligned to rows.
       Returns {k: acc}."""
    df = meta_df.copy()
    df["score"] = scores
    out = {k: 0 for k in ks}
    n_tr = 0
    for tid, sub in df.groupby("transition_id", sort=False):
        if sub["is_positive"].sum() == 0:
            continue
        sub2 = sub.sort_values("score", ascending=False)
        # rank of the first positive
        # use the rank of any positive (there should be 1)
        ranks = np.where(sub2["is_positive"].values == 1)[0]
        if len(ranks) == 0:
            continue
        r = int(ranks[0])
        for k in ks:
            if r < k:
                out[k] += 1
        n_tr += 1
    if n_tr == 0:
        return {f"top{k}": float("nan") for k in ks}, 0
    return {f"top{k}": out[k] / n_tr for k in ks}, n_tr


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0_all = time.time()
    LOG.info("=== Gate 6 baseline ===")
    LOG.info("Loading data...")
    transitions = pd.read_parquet(PROC / "transitions.parquet")
    trajectories = pd.read_parquet(PROC / "trajectories.parquet")
    dms = pd.read_parquet(PROC / "dms_fitness.parquet")
    md5_to_amr = pd.read_parquet(EMB / "_md5_to_amr.parquet")
    residue_index = pd.read_parquet(EMB / "residue_embeddings_index.parquet")
    protein_npz = np.load(EMB / "protein_embeddings.npz")
    LOG.info("Transitions: %d. Trajectories: %d. DMS: %d. md5_to_amr: %d. "
             "residues idx: %d. protein npz keys: %d",
             len(transitions), len(trajectories), len(dms), len(md5_to_amr),
             len(residue_index), len(protein_npz.files))

    # ---- attach organism to transitions (dedupe trajectory_id first)
    traj_dedup = trajectories.drop_duplicates(subset=["trajectory_id"], keep="first")
    tj_org = traj_dedup.set_index("trajectory_id")["organism"]
    trans = transitions.copy()
    trans["organism"] = trans["trajectory_id"].map(tj_org).fillna("unknown")
    LOG.info("Organism breakdown:\n%s", trans["organism"].value_counts().to_string())

    # ---- compute canonical AMR mutation candidates already present in trajectories
    canonical_muts_seen = set()
    for m in transitions["mutation_added"]:
        g, _, _ = parse_mut(m)
        if g in CANONICAL_AMR_GENES:
            canonical_muts_seen.add(m)
    LOG.info("Canonical AMR mutations in vocab: %d", len(canonical_muts_seen))

    # ---- build candidate vocabularies
    vocab = build_candidate_vocab(transitions, min_count_per_class=2,
                                  canonical_amr_mutations=canonical_muts_seen)
    for dc, v in vocab.items():
        LOG.info("Vocab[%s] = %d candidates", dc, len(v))

    # ---- coverage of true mutations in vocab
    in_vocab = 0
    for _, row in transitions.iterrows():
        v = vocab.get(row["drug_class"], [])
        if row["mutation_added"] in v:
            in_vocab += 1
    LOG.info("True mutation in vocab coverage: %d / %d = %.1f%%",
             in_vocab, len(transitions), in_vocab/len(transitions)*100)

    # ---- DMS lookup
    LOG.info("Building DMS lookup...")
    dms_idx = build_dms_lookup(dms)
    LOG.info("DMS index size: %d", len(dms_idx))

    # ---- gene_symbol -> md5
    gene_to_md5 = build_gene_to_md5(md5_to_amr)
    LOG.info("Distinct gene_symbols with md5: %d", len(gene_to_md5))

    # ---- Fit PCA on protein embeddings
    # NOTE: kept at 64 / 32 components per the original design, even though gain analysis
    # in the first run showed near-zero contribution. We save the PCA models so
    # the week-2 autoregressive predictor can swap them in at inference.
    pca_p_path = MODELS / "esm2_pca_protein.pkl"
    pca_r_path = MODELS / "esm2_pca_residue.pkl"
    if pca_p_path.exists():
        LOG.info("Loading cached protein PCA")
        with open(pca_p_path, "rb") as f:
            pca_p = pickle.load(f)
        # rebuild md5_to_pc
        keys = protein_npz.files
        X = np.zeros((len(keys), 1280), dtype=np.float32)
        for i, k in enumerate(keys):
            X[i] = protein_npz[k].astype(np.float32, copy=False)
        Xp = pca_p.transform(X)
        md5_to_pc = {k: Xp[i] for i, k in enumerate(keys)}
    else:
        pca_p, md5_to_pc = fit_protein_pca(protein_npz, n_components=64)
        with open(pca_p_path, "wb") as f:
            pickle.dump(pca_p, f)

    if pca_r_path.exists():
        LOG.info("Loading cached residue PCA")
        with open(pca_r_path, "rb") as f:
            pca_r = pickle.load(f)
    else:
        pca_r = fit_residue_pca(residue_index, EMB / "residue_embeddings",
                                n_components=32, sample_max=120000)
        with open(pca_r_path, "wb") as f:
            pickle.dump(pca_r, f)

    feat_builder = FeatureBuilder(
        protein_emb_pca=md5_to_pc,
        residue_emb_index=residue_index,
        residue_dir=EMB / "residue_embeddings",
        protein_npz=protein_npz,
        dms_idx=dms_idx,
        gene_to_md5=gene_to_md5,
        protein_pca_model=pca_p,
        residue_pca_model=pca_r,
        n_proteinpca=64,
        n_residpca=32,
    )

    # ---- Define test splits
    # Tag transitions for splits
    trans["split"] = "train"

    # ALEdb-124 sub-experiment holdout (within-FQ): e1226 (mid-size, 914 trans)
    # Plus a SECOND sub-experiment held out as the primary E.coli x cipro test:
    #   pick e1230 (880 transitions, ~46 trajectories) — disjoint from training,
    #   no shared starting strain with training.
    def get_exp_code(tid):
        m = re.match(r"^(?:norfloxacin)_(e\d+)_", tid)
        return m.group(1) if m else None
    trans["ale_exp"] = trans["trajectory_id"].apply(get_exp_code)
    ALE_HOLDOUT_EXP = "e1226"           # within-FQ secondary
    PRIMARY_ECO_FQ_HOLDOUT_EXP = "e1230"  # primary E.coli x FQ test
    trans.loc[trans["ale_exp"] == ALE_HOLDOUT_EXP, "split"] = "test_ale_fq_holdout"
    trans.loc[trans["ale_exp"] == PRIMARY_ECO_FQ_HOLDOUT_EXP, "split"] = "test_eco_fq"

    # S. aureus x oxacillin: small, only a handful of inferred trajectories
    is_sa_ox = (trans["organism"] == "Staphylococcus aureus") & \
               (trans["drug"].isin(["oxacillin", "methicillin"]))
    trans.loc[is_sa_ox & (trans["split"] == "train"), "split"] = "test_sa_oxa"

    # K. pneumoniae x meropenem
    is_kp_mero = (trans["organism"] == "Klebsiella pneumoniae") & \
                 (trans["drug"].isin(["meropenem", "imipenem", "ertapenem"]))
    trans.loc[is_kp_mero & (trans["split"] == "train"), "split"] = "test_kp_mero"

    LOG.info("Split sizes:\n%s", trans["split"].value_counts().to_string())

    # ---- Compute mutation frequency from train-split-only transitions
    train_trans_only = trans[trans["split"] == "train"]
    mut_freq_in_train = Counter(train_trans_only["mutation_added"].tolist())
    mut_freq_per_class = {}
    for dc, sub in train_trans_only.groupby("drug_class"):
        mut_freq_per_class[dc] = Counter(sub["mutation_added"].tolist())
    LOG.info("Train-only mut_freq: %d unique mutations", len(mut_freq_in_train))

    # ---- Pre-compute per-candidate feature cache
    cand_cache = precompute_candidate_features(
        vocab, feat_builder, dms_idx, gene_to_md5,
        mut_freq_in_train=mut_freq_in_train,
        mut_freq_per_class=mut_freq_per_class,
    )

    # ---- Build features
    LOG.info("Building features for train + test splits (max 50 negs per train row)")
    X_all, Y_all, G_all, META_all, feat_names = build_features(
        trans, vocab, feat_builder, dms_idx, gene_to_md5,
        max_neg_per_pos=50, split_col="split", cand_cache=cand_cache,
    )
    META_all = META_all.merge(trans[["transition_id", "split"]],
                              on="transition_id", how="left")
    LOG.info("X shape: %s; positive labels: %d/%d", X_all.shape,
             int(Y_all.sum()), len(Y_all))

    # Split index masks
    train_mask = META_all["split"].values == "train"
    test_masks = {
        "eco_fq": META_all["split"].values == "test_eco_fq",
        "ale_fq_holdout": META_all["split"].values == "test_ale_fq_holdout",
        "sa_oxa": META_all["split"].values == "test_sa_oxa",
        "kp_mero": META_all["split"].values == "test_kp_mero",
    }
    LOG.info("Rows per split: train=%d, eco_fq=%d, ale_fq=%d, sa_oxa=%d, kp_mero=%d",
             int(train_mask.sum()),
             int(test_masks["eco_fq"].sum()),
             int(test_masks["ale_fq_holdout"].sum()),
             int(test_masks["sa_oxa"].sum()),
             int(test_masks["kp_mero"].sum()))

    # ---- Build LightGBM rank dataset
    def make_group_array(meta_subset):
        # transition_id ordering preserved; count rows per transition
        # but require that rows are contiguous per transition
        # we'll sort first
        order = np.argsort(meta_subset["transition_id"].values, kind="stable")
        return order

    # Sort all rows by transition_id so groups are contiguous
    # Note: we sort within each split separately for inference
    LOG.info("Sorting rows by transition_id for grouping...")
    sort_order = np.argsort(META_all["transition_id"].values, kind="stable")
    X_all_s = X_all[sort_order]
    Y_all_s = Y_all[sort_order]
    META_all_s = META_all.iloc[sort_order].reset_index(drop=True)

    train_mask_s = META_all_s["split"].values == "train"
    test_masks_s = {
        "eco_fq": META_all_s["split"].values == "test_eco_fq",
        "ale_fq_holdout": META_all_s["split"].values == "test_ale_fq_holdout",
        "sa_oxa": META_all_s["split"].values == "test_sa_oxa",
        "kp_mero": META_all_s["split"].values == "test_kp_mero",
    }

    def group_sizes(meta):
        # rows per transition_id, in order
        s = meta.groupby("transition_id", sort=False).size().values
        return s

    train_meta = META_all_s[train_mask_s].reset_index(drop=True)
    train_X = X_all_s[train_mask_s]
    train_Y = Y_all_s[train_mask_s]
    train_groups = group_sizes(train_meta)
    LOG.info("Train transitions: %d  rows: %d", len(train_groups), int(train_mask_s.sum()))

    # ---- Train LightGBM LambdaRank with fixed hyperparameters
    # (We skip Optuna for simplicity in this run; given the large train set,
    # default LambdaRank hyperparameters work well. Hyperparameters are
    # documented below.)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [1, 3, 10],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 0.1,
        "verbose": -1,
        "max_depth": -1,
    }
    LOG.info("LightGBM params: %s", params)

    # Use 10% of train transitions as validation for early stopping
    train_tids = train_meta["transition_id"].unique()
    rng2 = np.random.default_rng(20260513)
    val_tids = set(rng2.choice(train_tids,
                               size=max(1, int(0.10 * len(train_tids))),
                               replace=False).tolist())
    val_mask = train_meta["transition_id"].isin(val_tids).values
    fit_mask = ~val_mask

    fit_X = train_X[fit_mask]
    fit_Y = train_Y[fit_mask]
    fit_meta = train_meta[fit_mask].reset_index(drop=True)
    fit_groups = group_sizes(fit_meta)

    val_X = train_X[val_mask]
    val_Y = train_Y[val_mask]
    val_meta = train_meta[val_mask].reset_index(drop=True)
    val_groups = group_sizes(val_meta)

    LOG.info("Fit transitions: %d  rows: %d ; Val transitions: %d rows: %d",
             len(fit_groups), len(fit_Y), len(val_groups), len(val_Y))

    ds_train = lgb.Dataset(fit_X, label=fit_Y, group=fit_groups,
                           feature_name=feat_names, free_raw_data=False)
    ds_val = lgb.Dataset(val_X, label=val_Y, group=val_groups,
                         feature_name=feat_names, free_raw_data=False)

    t_fit_0 = time.time()
    model = lgb.train(
        params,
        ds_train,
        num_boost_round=500,
        valid_sets=[ds_train, ds_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=25),
        ],
    )
    fit_time = time.time() - t_fit_0
    LOG.info("LightGBM training done. Best iter: %s  Time: %.1fs",
             model.best_iteration, fit_time)

    # ---- Save model
    with open(MODELS / "single_step_lgbm.pkl", "wb") as f:
        pickle.dump(model, f)

    # ---- Evaluation on each test split
    eval_results = {}
    for name, mask in test_masks_s.items():
        sub_X = X_all_s[mask]
        sub_meta = META_all_s[mask].reset_index(drop=True)
        n_trans = sub_meta["transition_id"].nunique()
        if n_trans == 0:
            LOG.info("Test split %s is empty", name)
            eval_results[name] = {"n_transitions": 0}
            continue
        scores = model.predict(sub_X)
        ks = (1, 3, 5, 10)
        accs, n_eval = topk_accuracy(sub_meta, scores, ks=ks)
        # Stratified breakdowns
        # First mutations (step==1) vs later
        first_meta = sub_meta[sub_meta["step_number"] == 1].reset_index(drop=True)
        first_scores = scores[(sub_meta["step_number"] == 1).values]
        later_meta = sub_meta[sub_meta["step_number"] > 1].reset_index(drop=True)
        later_scores = scores[(sub_meta["step_number"] > 1).values]

        # Canonical AMR breakdown by *true positive's canonical_amr_flag*
        # Map transition_id -> canonical_amr_flag
        tid_canon = sub_meta.drop_duplicates("transition_id").set_index("transition_id")["canonical_amr_flag"]
        canon_tids = set(tid_canon[tid_canon == 1].index.tolist())
        canon_mask = sub_meta["transition_id"].isin(canon_tids).values
        canon_meta = sub_meta[canon_mask].reset_index(drop=True)
        canon_scores = scores[canon_mask]
        passenger_meta = sub_meta[~canon_mask].reset_index(drop=True)
        passenger_scores = scores[~canon_mask]

        first_accs, n_first = topk_accuracy(first_meta, first_scores, ks=ks)
        later_accs, n_later = topk_accuracy(later_meta, later_scores, ks=ks)
        canon_accs, n_canon = topk_accuracy(canon_meta, canon_scores, ks=ks)
        passenger_accs, n_passenger = topk_accuracy(passenger_meta, passenger_scores, ks=ks)

        eval_results[name] = {
            "n_transitions_evaluated": n_eval,
            "all": accs,
            "step_first": {"n": n_first, **first_accs},
            "step_later": {"n": n_later, **later_accs},
            "canonical_amr_pos": {"n": n_canon, **canon_accs},
            "passenger_pos": {"n": n_passenger, **passenger_accs},
        }
        LOG.info("Eval[%s] all=%s n=%d", name, accs, n_eval)
        LOG.info("Eval[%s] first=%s later=%s canon=%s passenger=%s",
                 name, first_accs, later_accs, canon_accs, passenger_accs)

    # ---- Feature gains
    importances = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    pd.DataFrame(feat_imp, columns=["feature", "gain"]).to_csv(
        RESULTS / "single_step_features.csv", index=False)
    LOG.info("Top 15 features by gain:")
    for n, g in feat_imp[:15]:
        LOG.info("  %s: %.1f", n, g)

    # ---- Save eval JSON
    out = {
        "params": params,
        "best_iter": int(model.best_iteration) if model.best_iteration else None,
        "fit_time_sec": fit_time,
        "n_features": int(X_all.shape[1]),
        "n_train_transitions": int(len(fit_groups)),
        "n_val_transitions": int(len(val_groups)),
        "splits": eval_results,
        "vocabulary_sizes": {dc: len(v) for dc, v in vocab.items()},
        "true_mut_in_vocab_pct": float(in_vocab / len(transitions) * 100),
        "ale_holdout_experiment": ALE_HOLDOUT_EXP,
        "primary_eco_fq_holdout_experiment": PRIMARY_ECO_FQ_HOLDOUT_EXP,
        "top_features_by_gain": [
            {"feature": n, "gain": float(g)} for n, g in feat_imp[:30]
        ],
    }
    with open(RESULTS / "single_step_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    LOG.info("Wrote %s", RESULTS / "single_step_eval.json")

    # ---- Pass/fail decision
    eco_top3 = eval_results.get("eco_fq", {}).get("all", {}).get("top3", 0.0)
    eco_top1 = eval_results.get("eco_fq", {}).get("all", {}).get("top1", 0.0)
    ale_top3 = eval_results.get("ale_fq_holdout", {}).get("all", {}).get("top3", 0.0)
    msg_flags = []
    if eco_top1 is not None and eco_top1 > 0.70:
        msg_flags.append("LEAKAGE_FLAG_top1>0.70")
    if eco_top3 is not None and eco_top3 < 0.25:
        msg_flags.append("FEATURE_ERROR_FLAG_top3<0.25")

    gate_pass = "PASS" if (eco_top3 >= 0.40 and not msg_flags) else "ATTN"
    log_line = f"GATE_6: {gate_pass} [top1_eco_fq={eco_top1:.4f}] " \
               f"[top3_eco_fq={eco_top3:.4f}] " \
               f"[top3_within_ALE_holdout={ale_top3:.4f}]"
    if msg_flags:
        log_line += " [" + ",".join(msg_flags) + "]"
    LOG.info(log_line)

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(LOGS / "gates.log", "a") as f:
        f.write(f"{ts} {log_line}\n")

    LOG.info("Total wall-clock: %.1fs (%.1f min)",
             time.time() - t0_all, (time.time() - t0_all) / 60.0)
    return out, log_line


if __name__ == "__main__":
    main()
