"""
Reparse existing AMRFinderPlus TSVs and rebuild resistance_genes.parquet
with point-mutation extraction now correctly handling _-delimited symbols
like 'parC_S80F'. Reuses functions from gate2_card.py.
"""
import json
import time
from pathlib import Path

import pandas as pd

import gate2_card as g


def main():
    t0 = time.time()

    # Determine genomes-by-species from FASTA dir (cheap and authoritative)
    pairs_by_sp = {}
    for sp in g.SPECIES:
        fa_root = g.FASTA_DIR / sp
        gids = sorted([p.stem for p in fa_root.glob("*.faa")])
        pairs_by_sp[sp] = gids
        print(f"[{sp}] {len(gids)} genomes")

    # Phenotype evidence maps
    lab_resistant_by_sp = {}
    pheno_evidence = {}
    for sp in g.SPECIES:
        lr, em = g.load_pheno_resistance(sp)
        lab_resistant_by_sp[sp] = lr
        pheno_evidence.update(em)
        print(f"[{sp}] lab-resistant: {len(lr)}")

    # Catalogue
    catalogue = g.parse_card_variant_mutations()
    print(f"catalogue: {len(catalogue)} genes, {sum(len(v) for v in catalogue.values())} muts")

    # Parse
    print("Parsing TSVs …")
    df = g.parse_calls({sp: gids for sp, gids in pairs_by_sp.items()}, pheno_evidence)
    print(f"raw rows: {len(df)}")

    # Cross-reference
    df = g.cross_reference_mutations(df, catalogue)
    df.to_parquet(g.OUT_PARQUET, index=False)
    print(f"wrote {g.OUT_PARQUET}")

    # Point-mutation stats
    pm = df[df["point_mutations"].apply(lambda x: x is not None and len(x) > 0)]
    print(f"rows with point mutations: {len(pm)}")
    print(f"  known_resistance_mutation True : {int(pm['known_resistance_mutation'].sum())}")
    print(f"  known_resistance_mutation False: {int((~pm['known_resistance_mutation']).sum())}")

    # Gate stats
    summary = {"species": {}}
    overall_resistant_annotated = 0
    overall_resistant_total = 0
    total_genomes_processed = 0
    total_genomes_with_call = 0

    for sp, gids in pairs_by_sp.items():
        gids_set = set(gids)
        lab_resistant = lab_resistant_by_sp[sp] & gids_set
        sp_calls = df[df["species"] == sp]
        gids_with_call = set(sp_calls["genome_id"].unique().tolist())
        lab_res_with_call = lab_resistant & gids_with_call
        pct_call = (len(lab_res_with_call) / max(1, len(lab_resistant))) * 100.0

        summary["species"][sp] = {
            "n_genomes_processed":                    len(gids),
            "n_genomes_with_any_call":                len(gids_with_call),
            "pct_genomes_with_any_call":              round(len(gids_with_call) / max(1, len(gids)) * 100.0, 2),
            "n_phenotype_lab_resistant_in_processed": len(lab_resistant),
            "n_phenotype_lab_resistant_with_call":    len(lab_res_with_call),
            "pct_phenotype_resistant_with_call":      round(pct_call, 2),
        }
        overall_resistant_annotated += len(lab_res_with_call)
        overall_resistant_total += len(lab_resistant)
        total_genomes_processed += len(gids)
        total_genomes_with_call += len(gids_with_call)

    overall_pct = (overall_resistant_annotated / max(1, overall_resistant_total)) * 100.0

    summary["overall"] = {
        "total_genomes_processed":             total_genomes_processed,
        "total_genomes_with_call":             total_genomes_with_call,
        "overall_pct_genomes_with_call":       round(total_genomes_with_call / max(1, total_genomes_processed) * 100.0, 2),
        "overall_lab_resistant":               overall_resistant_total,
        "overall_lab_resistant_with_call":     overall_resistant_annotated,
        "overall_pct_lab_resistant_with_call": round(overall_pct, 2),
    }
    summary["top_element_symbols"]    = (df["gene_symbol"].value_counts().head(20)).to_dict()
    summary["amr_class_distribution"] = (df["amr_class"].value_counts().head(25)).to_dict()
    summary["point_mutations"] = {
        "n_call_rows_with_point_mutations": int(len(pm)),
        "n_known_resistance":               int(pm["known_resistance_mutation"].sum()),
        "n_novel_or_unmapped":              int((~pm["known_resistance_mutation"]).sum()),
    }

    # evidence tier breakdown
    evid_call_rate = {}
    processed_set = {gid for gids in pairs_by_sp.values() for gid in gids}
    for tier in ["lab", "computational", "mixed", "unknown"]:
        gids_in_tier = {gid for (sp, gid), t in pheno_evidence.items() if t == tier}
        in_tier_processed = gids_in_tier & processed_set
        with_call = in_tier_processed & set(df["genome_id"].unique().tolist())
        evid_call_rate[tier] = {
            "n_processed":   len(in_tier_processed),
            "n_with_call":   len(with_call),
            "call_rate_pct": round(len(with_call) / max(1, len(in_tier_processed)) * 100.0, 2),
        }
    summary["evidence_tier_call_rate"] = evid_call_rate

    # Top gene families (consolidated)
    # group by first 5 chars of gene_symbol (e.g. blaTEM-1 -> blaTE-, gyrA_S83L -> gyrA_)
    def fam(sym: str) -> str:
        if not isinstance(sym, str): return ""
        return sym.split("_")[0].split("-")[0]
    df["_family"] = df["gene_symbol"].apply(fam)
    summary["top_gene_families"] = df["_family"].value_counts().head(20).to_dict()

    # Known vs novel POINT calls
    point_calls = df[df["method"].str.startswith("POINT", na=False)]
    summary["point_call_calls"] = {
        "n_point_method_rows":     int(len(point_calls)),
        "n_with_extracted_muts":   int(point_calls["point_mutations"].apply(lambda x: len(x) > 0).sum()),
        "n_known_resistance":      int(point_calls["known_resistance_mutation"].sum()),
        "n_novel_or_unmapped":     int((~point_calls["known_resistance_mutation"]).sum()),
    }

    summary["card_catalogue_size"] = {
        "genes":     len(catalogue),
        "mutations": sum(len(v) for v in catalogue.values()),
    }
    summary["amrfinder_version"]    = "4.2.7"
    summary["amrfinder_db_version"] = "2026-03-24.1"
    summary["card_release"]         = "broadstreet_2025-05-29"
    summary["wall_clock_seconds"]   = round(time.time() - t0, 1)

    g.SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {g.SUMMARY_JSON}")

    # Re-emit Gate 2 log line
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    verdict = "PASS" if overall_pct >= 60.0 else ("PASS_SOFT_FLAG" if overall_pct >= 40.0 else "HARD_STOP")
    gate_line = (
        f"{ts} GATE_2_REPARSE: {verdict} "
        f"[overall_pct_resistant_with_call={overall_pct:.2f}%] "
        f"[ecoli={summary['species']['ecoli']['pct_phenotype_resistant_with_call']}%] "
        f"[saureus={summary['species']['saureus']['pct_phenotype_resistant_with_call']}%] "
        f"[kpneumo={summary['species']['kpneumo']['pct_phenotype_resistant_with_call']}%] "
        f"[point_muts_extracted={summary['point_call_calls']['n_with_extracted_muts']}] "
        f"[known_muts={summary['point_call_calls']['n_known_resistance']}]"
    )
    with g.GATE_LOG.open("a") as fh:
        fh.write(gate_line + "\n")
    print(gate_line)


if __name__ == "__main__":
    main()
