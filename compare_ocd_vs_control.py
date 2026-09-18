"""
Statistical comparison of OCD vs Control TEA network metrics.
Mann-Whitney U tests, effect sizes (Cohen's d), and summary tables.
"""
import csv
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(__file__)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def effect_label(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    ocd = pd.read_csv(os.path.join(HERE, "results", "tea_metrics.csv"))
    ctrl = pd.read_csv(os.path.join(HERE, "control_gutenberg", "results", "tea_metrics.csv"))

    ocd["group"] = "OCD"
    ctrl["group"] = "Control"

    # Normalize metrics by text length where appropriate
    for df in [ocd, ctrl]:
        df["triplets_per_word"] = df["n_triplets"] / df["n_words"]
        df["nodes_per_word"] = df["n_nodes"] / df["n_words"]
        df["edges_per_word"] = df["n_edges"] / df["n_words"]

    metrics = [
        ("first_person_ratio", "First-Person Agent Ratio"),
        ("repetition_idx", "Edge Repetition Index"),
        ("node_ttr", "Node Type-Token Ratio"),
        ("lscc_frac", "LSCC Fraction"),
        ("avg_clustering", "Avg Clustering Coefficient"),
        ("density", "Graph Density"),
        ("mean_edge_weight", "Mean Edge Weight"),
        ("triplets_per_word", "TEA Triplets per Word"),
        ("nodes_per_word", "Unique Nodes per Word"),
    ]

    print("=" * 90)
    print(f"{'Metric':35s} {'OCD (M±SD)':18s} {'Ctrl (M±SD)':18s} {'U':>10s} {'p':>10s} {'d':>8s} {'Effect':>12s}")
    print("=" * 90)

    results = []
    for col, label in metrics:
        ocd_vals = ocd[col].dropna().values
        ctrl_vals = ctrl[col].dropna().values

        ocd_m, ocd_s = np.mean(ocd_vals), np.std(ocd_vals)
        ctrl_m, ctrl_s = np.mean(ctrl_vals), np.std(ctrl_vals)

        u_stat, p_val = stats.mannwhitneyu(ocd_vals, ctrl_vals, alternative="two-sided")
        d = cohens_d(ocd_vals, ctrl_vals)
        eff = effect_label(d)

        sig = ""
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"

        print(f"  {label:33s} {ocd_m:.4f}±{ocd_s:.4f}  {ctrl_m:.4f}±{ctrl_s:.4f}  "
              f"{u_stat:10.0f} {p_val:10.6f}{sig:3s} {d:+8.3f} {eff:>12s}")

        results.append({
            "metric": label,
            "ocd_mean": round(ocd_m, 4),
            "ocd_std": round(ocd_s, 4),
            "ctrl_mean": round(ctrl_m, 4),
            "ctrl_std": round(ctrl_s, 4),
            "U": u_stat,
            "p_value": round(p_val, 6),
            "cohens_d": round(d, 3),
            "effect_size": eff,
            "significant": p_val < 0.05,
        })

    print("=" * 90)
    print(f"\n  OCD corpus: n={len(ocd)}, Control corpus: n={len(ctrl)}")
    print(f"  Significance: * p<0.05, ** p<0.01, *** p<0.001")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    sig_results = [r for r in results if r["significant"]]
    for r in sorted(sig_results, key=lambda x: abs(x["cohens_d"]), reverse=True):
        direction = "higher" if r["cohens_d"] > 0 else "lower"
        print(f"  {r['metric']:35s} OCD is {direction} (d={r['cohens_d']:+.3f}, {r['effect_size']}, p={r['p_value']:.6f})")

    # Save results
    out_csv = os.path.join(HERE, "results", "ocd_vs_control_comparison.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  Results saved to {out_csv}")

    # Additional: word count comparison (to check if groups are matched)
    print(f"\n  Word count comparison:")
    print(f"    OCD:     mean={ocd['n_words'].mean():.0f}, median={ocd['n_words'].median():.0f}")
    print(f"    Control: mean={ctrl['n_words'].mean():.0f}, median={ctrl['n_words'].median():.0f}")
    u, p = stats.mannwhitneyu(ocd["n_words"], ctrl["n_words"], alternative="two-sided")
    d_wc = cohens_d(ocd["n_words"].values, ctrl["n_words"].values)
    print(f"    U={u:.0f}, p={p:.6f}, d={d_wc:+.3f}")
    if p < 0.05:
        print("    WARNING: Groups differ significantly in word count -- normalized metrics are more reliable")


if __name__ == "__main__":
    main()
