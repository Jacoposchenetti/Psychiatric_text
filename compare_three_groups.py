"""
Three-group comparison: OCD Stories vs Cancer Patient Stories vs Depression/PTSD Reddit.
All comparisons length-matched.
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
        return "negl."
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "LARGE"


def match_by_word_count(group_a, group_b, tolerance=0.3):
    np.random.seed(42)
    b_avail = group_b.copy().reset_index(drop=True)
    b_avail["_matched"] = False

    matched_a = []
    matched_b = []

    a_shuffled = group_a.sample(frac=1, random_state=42).reset_index(drop=True)

    for _, row_a in a_shuffled.iterrows():
        wc_a = row_a["n_words"]
        avail = b_avail[~b_avail["_matched"]]
        if avail.empty:
            break

        diffs = (avail["n_words"] - wc_a).abs()
        best_idx = diffs.idxmin()
        best_diff = diffs[best_idx]

        if best_diff / max(wc_a, 1) <= tolerance:
            b_avail.loc[best_idx, "_matched"] = True
            matched_a.append(row_a)
            matched_b.append(b_avail.loc[best_idx])

    return pd.DataFrame(matched_a), pd.DataFrame(matched_b)


def compare_pair(ocd, ctrl, ctrl_name, metrics):
    ocd_m, ctrl_m = match_by_word_count(ocd, ctrl, tolerance=0.35)

    if len(ocd_m) < 20:
        print(f"  Too few matched pairs ({len(ocd_m)}) for {ctrl_name}")
        return []

    u_wc, p_wc = stats.mannwhitneyu(ocd_m["n_words"], ctrl_m["n_words"], alternative="two-sided")
    d_wc = cohens_d(ocd_m["n_words"].values, ctrl_m["n_words"].values)

    print(f"\n  OCD vs {ctrl_name}: {len(ocd_m)} matched pairs")
    print(f"  OCD words:  mean={ocd_m['n_words'].mean():.0f}, median={ocd_m['n_words'].median():.0f}")
    print(f"  Ctrl words: mean={ctrl_m['n_words'].mean():.0f}, median={ctrl_m['n_words'].median():.0f}")
    print(f"  Word count match: d={d_wc:+.3f} ({effect_label(d_wc)}), p={p_wc:.4f}")

    print(f"\n  {'Metric':35s} {'OCD':>10s} {'Ctrl':>10s} {'d':>8s} {'p':>10s} {'Eff':>8s}")
    print("  " + "-" * 85)

    results = []
    for col, label in metrics:
        ocd_vals = ocd_m[col].dropna().values
        ctrl_vals = ctrl_m[col].dropna().values

        if len(ocd_vals) < 10 or len(ctrl_vals) < 10:
            continue

        u, p = stats.mannwhitneyu(ocd_vals, ctrl_vals, alternative="two-sided")
        d = cohens_d(ocd_vals, ctrl_vals)
        eff = effect_label(d)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"  {label:35s} {np.mean(ocd_vals):10.4f} {np.mean(ctrl_vals):10.4f} "
              f"{d:+8.3f} {p:10.6f}{sig:3s} {eff:>8s}")

        results.append({
            "comparison": f"OCD vs {ctrl_name}",
            "metric": label,
            "ocd_mean": round(np.mean(ocd_vals), 4),
            "ctrl_mean": round(np.mean(ctrl_vals), 4),
            "cohens_d": round(d, 3),
            "p_value": round(p, 6),
            "effect": eff,
            "n_pairs": len(ocd_m),
            "significant": p < 0.05,
        })

    return results


def main():
    ocd = pd.read_csv(os.path.join(HERE, "results", "tea_metrics.csv"))
    cancer = pd.read_csv(os.path.join(HERE, "control_patient_stories", "results", "tea_metrics.csv"))
    mh = pd.read_csv(os.path.join(HERE, "control_mental_health", "results", "tea_metrics.csv"))

    for df in [ocd, cancer, mh]:
        df["triplets_per_word"] = df["n_triplets"] / df["n_words"]
        df["nodes_per_word"] = df["n_nodes"] / df["n_words"]

    print("=" * 100)
    print("  THREE-GROUP COMPARISON (length-matched)")
    print("  OCD Stories (blog) vs Cancer Stories (blog) vs Depression/PTSD (Reddit)")
    print("=" * 100)
    print(f"\n  Group sizes before matching:")
    print(f"    OCD Stories:      n={len(ocd)}, mean words={ocd['n_words'].mean():.0f}")
    print(f"    Cancer Stories:   n={len(cancer)}, mean words={cancer['n_words'].mean():.0f}")
    print(f"    Depression/PTSD:  n={len(mh)}, mean words={mh['n_words'].mean():.0f}")

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

    all_results = []

    # OCD vs Cancer
    print("\n" + "=" * 100)
    all_results.extend(compare_pair(ocd, cancer, "Cancer", metrics))

    # OCD vs Depression/PTSD
    print("\n" + "=" * 100)
    all_results.extend(compare_pair(ocd, mh, "Depression/PTSD", metrics))

    # Summary
    print("\n" + "=" * 100)
    print("  SUMMARY: Significant findings across both comparisons")
    print("=" * 100)

    sig = [r for r in all_results if r["significant"]]

    # Group by metric
    from collections import defaultdict
    by_metric = defaultdict(list)
    for r in sig:
        by_metric[r["metric"]].append(r)

    for metric, findings in sorted(by_metric.items(), key=lambda x: -max(abs(f["cohens_d"]) for f in x[1])):
        print(f"\n  {metric}:")
        for f in findings:
            dir_ = "HIGHER" if f["cohens_d"] > 0 else "LOWER"
            print(f"    vs {f['comparison'].split('vs ')[1]:20s}: OCD {dir_} d={f['cohens_d']:+.3f} ({f['effect']}, p={f['p_value']:.6f}, n={f['n_pairs']})")

    # Save
    if all_results:
        out_csv = os.path.join(HERE, "results", "three_group_comparison.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\n  Saved to {out_csv}")


if __name__ == "__main__":
    main()
