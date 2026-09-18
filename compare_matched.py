"""
Matched comparison: OCD vs Patient Story control, controlling for text length.
Uses propensity-score-like matching on word count.
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
        return "LARGE"


def match_by_word_count(ocd, ctrl, tolerance=0.3):
    """Match each OCD story with the closest-length control story (without replacement)."""
    np.random.seed(42)
    ctrl_available = ctrl.copy().reset_index(drop=True)
    ctrl_available["_matched"] = False

    matched_ocd = []
    matched_ctrl = []

    ocd_shuffled = ocd.sample(frac=1, random_state=42).reset_index(drop=True)

    for _, ocd_row in ocd_shuffled.iterrows():
        ocd_wc = ocd_row["n_words"]
        avail = ctrl_available[~ctrl_available["_matched"]]
        if avail.empty:
            break

        diffs = (avail["n_words"] - ocd_wc).abs()
        best_idx = diffs.idxmin()
        best_diff = diffs[best_idx]

        # Only match if within tolerance (relative)
        if best_diff / max(ocd_wc, 1) <= tolerance:
            ctrl_available.loc[best_idx, "_matched"] = True
            matched_ocd.append(ocd_row)
            matched_ctrl.append(ctrl_available.loc[best_idx])

    return pd.DataFrame(matched_ocd), pd.DataFrame(matched_ctrl)


def main():
    ocd_all = pd.read_csv(os.path.join(HERE, "results", "tea_metrics.csv"))
    ctrl_all = pd.read_csv(os.path.join(HERE, "control_patient_stories", "results", "tea_metrics.csv"))

    for df in [ocd_all, ctrl_all]:
        df["triplets_per_word"] = df["n_triplets"] / df["n_words"]
        df["nodes_per_word"] = df["n_nodes"] / df["n_words"]
        df["edges_per_word"] = df["n_edges"] / df["n_words"]

    print("=" * 100)
    print("  WORD-COUNT MATCHED COMPARISON: OCD vs Patient Story Control")
    print("=" * 100)

    ocd, ctrl = match_by_word_count(ocd_all, ctrl_all, tolerance=0.3)
    print(f"\n  Matched pairs: {len(ocd)}")
    print(f"  OCD word count:  mean={ocd['n_words'].mean():.0f}, median={ocd['n_words'].median():.0f}, SD={ocd['n_words'].std():.0f}")
    print(f"  Ctrl word count: mean={ctrl['n_words'].mean():.0f}, median={ctrl['n_words'].median():.0f}, SD={ctrl['n_words'].std():.0f}")
    u_wc, p_wc = stats.mannwhitneyu(ocd["n_words"], ctrl["n_words"], alternative="two-sided")
    d_wc = cohens_d(ocd["n_words"].values, ctrl["n_words"].values)
    print(f"  Word count match: U={u_wc:.0f}, p={p_wc:.4f}, d={d_wc:+.3f} ({effect_label(d_wc)})")

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

    print(f"\n  {'Metric':35s} {'OCD (M+/-SD)':18s} {'Ctrl (M+/-SD)':18s} {'U':>10s} {'p':>10s} {'d':>8s} {'Effect':>12s}")
    print("-" * 100)

    results = []
    for col, label in metrics:
        ocd_vals = ocd[col].dropna().values
        ctrl_vals = ctrl[col].dropna().values

        if len(ocd_vals) < 5 or len(ctrl_vals) < 5:
            print(f"  {label:33s} insufficient data")
            continue

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

        print(f"  {label:33s} {ocd_m:.4f}+/-{ocd_s:.4f}  {ctrl_m:.4f}+/-{ctrl_s:.4f}  "
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

    print("=" * 100)

    # Key findings
    print("\n  KEY FINDINGS (length-matched)")
    print("=" * 60)

    sig_results = [r for r in results if r["significant"]]
    for r in sorted(sig_results, key=lambda x: abs(x["cohens_d"]), reverse=True):
        direction = "HIGHER" if r["cohens_d"] > 0 else "LOWER"
        print(f"  OCD {direction:6s}: {r['metric']:35s} d={r['cohens_d']:+.3f} ({r['effect_size']}, p={r['p_value']:.6f})")

    ns = [r for r in results if not r["significant"]]
    if ns:
        print(f"\n  Not significant:")
        for r in ns:
            print(f"    {r['metric']:35s} d={r['cohens_d']:+.3f} (p={r['p_value']:.3f})")

    # Save
    out_csv = os.path.join(HERE, "results", "ocd_vs_patient_matched_comparison.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  Saved to {out_csv}")


if __name__ == "__main__":
    main()
