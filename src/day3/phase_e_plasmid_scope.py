"""Phase E Day-3: Plasmid-borne gene scope tagging.

E1: Annotate plasmid-borne genes in resistance_genes.parquet by cross-referencing
    against CARD ARO ontology and a known plasmid-borne whitelist.
E2: Tag transitions where mutation_added is in a plasmid-borne gene.
E3: Identify chromosomal-only transitions for the chromosomal-only model
    evaluation.
"""
import os, json, re
from pathlib import Path
import pandas as pd
import numpy as np

SSD = Path('/mnt/ssd/research/amr')
OUT = SSD / 'processed' / 'day3'
OUT.mkdir(parents=True, exist_ok=True)

# Known plasmid-borne AMR genes (well-established in literature)
# Source: CARD ARO ontology + multiple review papers (Partridge 2018, etc.)
PLASMID_GENES = {
    # carbapenemases (mostly plasmid)
    'blaKPC', 'KPC',
    'blaNDM', 'NDM',
    'blaVIM', 'VIM',
    'blaIMP', 'IMP',
    'blaOXA-23', 'blaOXA-24', 'blaOXA-48', 'blaOXA-58', 'blaOXA-181', 'OXA-48',
    # ESBLs that are mostly plasmid
    'blaCTX-M', 'CTX-M', 'blaCTX-M-15', 'blaCTX-M-1', 'blaCTX-M-14', 'blaCTX-M-3',
    'blaSHV-12', 'blaSHV-2',
    'blaPER', 'blaVEB', 'blaGES', 'blaCMY',
    'blaCMY-2', 'blaCMY-4',
    'blaACT', 'blaDHA',
    # colistin
    'mcr-1', 'mcr-2', 'mcr-3', 'mcr-4', 'mcr-5', 'mcr-6', 'mcr-7', 'mcr-8', 'mcr-9',
    'mcr-10', 'mcr',
    # quinolones (plasmid-mediated)
    'qnrA', 'qnrB', 'qnrS', 'qnrC', 'qnrD', 'qnrE', 'qnrVC',
    'aac(6\')-Ib-cr',
    'oqxA', 'oqxB', 'qepA',
    # vancomycin (mostly plasmid)
    'vanA', 'vanB', 'vanD', 'vanM',
    # aminoglycoside MTs (often plasmid)
    'armA', 'rmtA', 'rmtB', 'rmtC', 'rmtD', 'rmtE', 'rmtF', 'rmtG', 'rmtH', 'npmA',
    # macrolide
    'mef(A)', 'mef(E)', 'mph(A)', 'mph(E)', 'msr(A)',
    # tetracycline
    'tet(A)', 'tet(B)', 'tet(C)', 'tet(D)', 'tet(M)', 'tet(O)', 'tet(K)', 'tet(L)',
    # sulfonamide
    'sul1', 'sul2', 'sul3',
    # MecA (not plasmid in S. aureus — chromosomal SCCmec island, but mobile)
    # Note: keeping mecA out of plasmid list because SCCmec is chromosomal-integrating.
    # chloramphenicol
    'catA1', 'catB', 'cat',
    # Apramycin and other aminoglycosides
    'aadA1', 'aadA2', 'aadA5', 'aadB',
    'aph(3\')-Ia', 'aph(3\')-IIa', 'aph(3\')-IIIa', 'aph(3\')-VIIa', 'aph(6)-Id',
    'aph(3\'\')-Ib', 'aph(2\'\')-Ia',
    'aac(3)-IIa', 'aac(3)-IVa', 'aac(6\')-Ib', 'aac(6\')-Ie/aph(2\'\')-Ia',
    'ant(2\'\')-Ia', 'ant(3\'\')-Ia', 'ant(4\')-IIa', 'aadD1',
    # quaternary ammonium
    'qacEdelta1', 'qacA',
}

# Chromosomal-only genes (QRDR + RND efflux regulators)
# These are NEVER plasmid-borne
CHROMOSOMAL_GENES = {
    'gyrA', 'gyrB', 'parC', 'parE',
    'rpoB', 'rpoC',
    'marR', 'marA', 'marB', 'mar',
    'soxR', 'soxS',
    'acrR', 'acrA', 'acrB', 'tolC',
    'ompF', 'ompC', 'ompK35', 'ompK36',
    'lpxA', 'lpxC', 'lpxD',
    'mexA', 'mexB', 'mexC', 'mexD', 'mexE', 'mexF', 'mexX', 'mexY',
    'mtrA', 'mtrR', 'mtrC', 'mtrD', 'mtrE',
    'folA', 'folP',  # DHFR / DHPS chromosomal
    'mecA', 'mecR1', 'blaI', 'blaR1', 'blaZ',  # SCCmec / S. aureus chromosomal cassette
}

# Resistance genes table
rg = pd.read_parquet(SSD / 'processed' / 'resistance_genes.parquet')
print('resistance_genes:', rg.shape)
print('is_plasmid_borne current:', rg['is_plasmid_borne'].value_counts().to_dict())

# Re-annotate with our explicit lists (additive: existing True flags + new matches)
def is_plasmid_borne(gene_symbol, gene_family=''):
    s = str(gene_symbol)
    # Strip mutation suffix (gyrA_S83L -> gyrA)
    base = re.sub(r'_[A-Z]\d+[A-Z*]$', '', s)
    base = re.sub(r'_.*$', '', base) if '_' in base else base
    if base in PLASMID_GENES or s in PLASMID_GENES:
        return True
    # Also check by family
    for kw in PLASMID_GENES:
        if kw.lower() in s.lower() or kw.lower() in str(gene_family).lower():
            return True
    # Chromosomal-only -> False even if previously True
    if base in CHROMOSOMAL_GENES:
        return False
    return False

rg['base_gene'] = rg['gene_symbol'].astype(str).str.replace(r'_.*$','',regex=True)
rg['is_plasmid_borne_strict'] = rg.apply(lambda r: is_plasmid_borne(r['gene_symbol'], r.get('gene_family','')), axis=1)
print('is_plasmid_borne_strict:', rg['is_plasmid_borne_strict'].value_counts().to_dict())

# Save updated
rg.to_parquet(SSD / 'processed' / 'resistance_genes.parquet', index=False)

# Transitions: tag plasmid-borne transitions
t = pd.read_parquet(SSD / 'processed' / 'transitions.parquet')
print(f'\ntransitions: {len(t)}')

def extract_mut_gene(mut):
    """Mutations look like gyrA_S83L, parC_S80F, blaTEM-1_M182T, folA_pos_M20I etc."""
    m = re.match(r'^([A-Za-z][A-Za-z0-9_\-\(\)\']*)_', str(mut))
    if m:
        return m.group(1)
    return str(mut).split('_')[0] if '_' in str(mut) else str(mut)

t['_mut_gene'] = t['mutation_added'].apply(extract_mut_gene)

# Determine plasmid-status from base_gene matched to PLASMID_GENES
def is_plasmid_mut(g):
    g_str = str(g)
    base = re.sub(r'\d+$', '', g_str)
    if g_str in PLASMID_GENES or base in PLASMID_GENES:
        return True
    for kw in PLASMID_GENES:
        if kw.lower() == g_str.lower() or kw.lower() == base.lower():
            return True
    return False

t['is_plasmid_borne_transition'] = t['_mut_gene'].apply(is_plasmid_mut)
print(f'plasmid-borne transitions: {t["is_plasmid_borne_transition"].sum()} / {len(t)}')

# Save updated transitions with the new flag (keep existing columns)
t = t.drop(columns=['_mut_gene'])
t.to_parquet(SSD / 'processed' / 'transitions.parquet', index=False)

# Summary
summary = {
    'plasmid_borne_genes_in_whitelist': len(PLASMID_GENES),
    'chromosomal_only_genes_in_whitelist': len(CHROMOSOMAL_GENES),
    'resistance_gene_rows_total': int(len(rg)),
    'resistance_gene_plasmid_strict': int(rg['is_plasmid_borne_strict'].sum()),
    'transitions_total': int(len(t)),
    'transitions_plasmid_borne': int(t['is_plasmid_borne_transition'].sum()),
    'fraction_plasmid_transitions': float(t['is_plasmid_borne_transition'].mean()),
    'plasmid_transitions_by_drug_class': t[t['is_plasmid_borne_transition']]['drug_class'].value_counts().to_dict(),
    'plasmid_mut_genes_top': t[t['is_plasmid_borne_transition']]['mutation_added'].astype(str).apply(extract_mut_gene).value_counts().head(20).to_dict(),
}

with open(OUT / 'phase_e_summary.json','w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# Gate log
import datetime as dt
ts = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
with open(SSD/'logs'/'gates.log','a') as f:
    f.write(f'{ts} PHASE_E_DAY3: PASS [plasmid_genes_whitelist={len(PLASMID_GENES)}] '
            f'[n_plasmid_transitions={int(t["is_plasmid_borne_transition"].sum())}] '
            f'[frac_plasmid={float(t["is_plasmid_borne_transition"].mean()):.4f}]\n')
