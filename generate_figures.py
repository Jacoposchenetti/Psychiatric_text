"""
Generate publication-quality figures for the OCD cognitive network analysis.
Uses Massimo Stella's TEA_Networks visualization functions (plot_svo_graph,
tea_weighted_degree_centrality) alongside custom comparison charts.

Figures produced:
  1. TEA network plots for sample stories (one per group)
  2. Three-group TEA metric comparison (bar + error bars)
  3. Three-group RQA metric comparison
  4. Cohen's d effect size forest plot
  5. Recurrence matrix heatmaps for sample texts
  6. Top-agent degree centrality comparison
"""
import sys
import os
import csv
import warnings
import random

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import spacy

sys.path.insert(0, r"C:\Users\jsche\Desktop\NLP\TEA_Networks")
from teanets.svo_extraction import extract_svos
from teanets.teaplot import plot_svo_graph
from teanets.analytics import (
    svo_to_graph,
    tea_weighted_degree_centrality,
    filter_svo_dataframe_by_tea,
)

random.seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {
    "OCD": "#E64B35",
    "Cancer": "#4DBBD5",
    "Dep/PTSD": "#00A087",
}

nlp = None

def load_nlp():
    global nlp
    if nlp is None:
        print("Loading en_core_web_lg...")
        nlp = spacy.load("en_core_web_lg")
        nlp.max_length = 50000
    return nlp


def pick_sample(meta_path, texts_dir, target_wc=1000, n=1):
    meta = []
    with open(meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["word_count"] = int(row["word_count"])
            meta.append(row)
    meta = [r for r in meta if r["word_count"] >= 500]
    meta.sort(key=lambda r: abs(r["word_count"] - target_wc))
    picked = []
    for row in meta[:n * 3]:
        txt = os.path.join(texts_dir, f"{row['slug']}.txt")
        if os.path.exists(txt):
            with open(txt, encoding="utf-8") as f:
                text = f.read()
            if len(text.split()) >= 300:
                picked.append((row["slug"], text, row))
                if len(picked) >= n:
                    break
    return picked


def extract_tea(text):
    _nlp = load_nlp()
    if len(text) > 30000:
        text = text[:30000]
    doc = _nlp(text)
    df = extract_svos(doc, semantic_relations=False)
    return df


# ── Figure 1: TEA Network Plots ──────────────────────────────────────────────

def fig1_tea_networks():
    print("\n=== Figure 1: TEA Network Plots (Stella's plot_svo_graph) ===")

    samples = {
        "OCD": pick_sample(
            os.path.join(HERE, "metadata.csv"),
            os.path.join(HERE, "texts"),
            target_wc=800, n=1
        ),
        "Cancer": pick_sample(
            os.path.join(HERE, "control_patient_stories", "metadata.csv"),
            os.path.join(HERE, "control_patient_stories", "texts"),
            target_wc=800, n=1
        ),
        "Dep/PTSD": pick_sample(
            os.path.join(HERE, "control_mental_health", "metadata.csv"),
            os.path.join(HERE, "control_mental_health", "texts"),
            target_wc=800, n=1
        ),
    }

    for group, items in samples.items():
        if not items:
            print(f"  No sample found for {group}")
            continue
        slug, text, meta = items[0]
        wc = len(text.split())
        print(f"  [{group}] {slug} ({wc} words)")

        df = extract_tea(text)
        if df is None or len(df) == 0:
            print(f"    No TEA triplets extracted")
            continue
        print(f"    {len(df)} TEA triplets extracted")

        safe_group = group.replace("/", "-")
        fname = os.path.join(FIG_DIR, f"tea_network_{safe_group}.png")
        plot_svo_graph(df, custom_font=10, filename=fname, seed=42, show=False)
        print(f"    Saved {fname}")

        fname_i = os.path.join(FIG_DIR, f"tea_network_{safe_group}_I.png")
        plot_svo_graph(df, subject_filter="I", custom_font=12,
                       filename=fname_i, seed=42, show=False)
        print(f"    Saved {fname_i} (filtered: subject='I')")


# ── Figure 2: TEA Metrics Bar Chart ──────────────────────────────────────────

def fig2_tea_comparison():
    print("\n=== Figure 2: TEA Metrics Three-Group Comparison ===")

    ocd = pd.read_csv(os.path.join(HERE, "results", "tea_metrics.csv"))
    cancer = pd.read_csv(os.path.join(HERE, "control_patient_stories", "results", "tea_metrics.csv"))
    mh = pd.read_csv(os.path.join(HERE, "control_mental_health", "results", "tea_metrics.csv"))

    metrics = [
        ("first_person_ratio", "First-Person\nAgent Ratio"),
        ("lscc_frac", "LSCC\nFraction"),
        ("density", "Graph\nDensity"),
        ("avg_clustering", "Avg\nClustering"),
        ("node_ttr", "Node\nType-Token Ratio"),
        ("repetition_idx", "Edge\nRepetition Index"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        groups = {"OCD": ocd, "Cancer": cancer, "Dep/PTSD": mh}
        means, stds, colors = [], [], []
        for name, df in groups.items():
            vals = df[col].dropna()
            means.append(vals.mean())
            stds.append(vals.std() / np.sqrt(len(vals)))
            colors.append(PALETTE[name])

        x = np.arange(3)
        bars = ax.bar(x, means, yerr=stds, width=0.6, color=colors,
                      edgecolor="white", linewidth=1.2, capsize=5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(list(groups.keys()), fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("TEA Network Metrics: OCD vs Cancer vs Depression/PTSD",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, "tea_metrics_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Figure 3: RQA Metrics Bar Chart ──────────────────────────────────────────

def fig3_rqa_comparison():
    print("\n=== Figure 3: RQA Metrics Three-Group Comparison ===")

    ocd = pd.read_csv(os.path.join(HERE, "results", "rqa_ocd.csv"))
    cancer = pd.read_csv(os.path.join(HERE, "results", "rqa_cancer.csv"))
    mh = pd.read_csv(os.path.join(HERE, "results", "rqa_mh.csv"))

    metrics = [
        ("RR", "Recurrence\nRate"),
        ("DET", "Determinism"),
        ("LAM", "Laminarity"),
        ("TT", "Trapping\nTime"),
        ("ENTR", "Recurrence\nEntropy"),
        ("L", "Mean Diagonal\nLength"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        groups = {"OCD": ocd, "Cancer": cancer, "Dep/PTSD": mh}
        means, stds, colors = [], [], []
        for name, df in groups.items():
            vals = df[col].dropna()
            means.append(vals.mean())
            stds.append(vals.std() / np.sqrt(len(vals)))
            colors.append(PALETTE[name])

        x = np.arange(3)
        bars = ax.bar(x, means, yerr=stds, width=0.6, color=colors,
                      edgecolor="white", linewidth=1.2, capsize=5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(list(groups.keys()), fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("RQA Metrics: OCD vs Cancer vs Depression/PTSD",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, "rqa_metrics_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Figure 4: Cohen's d Forest Plot ──────────────────────────────────────────

def fig4_effect_sizes():
    print("\n=== Figure 4: Cohen's d Effect Size Forest Plot ===")

    tea_comp = pd.read_csv(os.path.join(HERE, "results", "three_group_comparison.csv"))
    rqa_comp = pd.read_csv(os.path.join(HERE, "results", "rqa_three_group_comparison.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharey=False)

    # TEA effects
    for comp_name, color, marker in [
        ("OCD vs Cancer", "#E64B35", "o"),
        ("OCD vs Depression/PTSD", "#00A087", "s"),
    ]:
        subset = tea_comp[tea_comp["comparison"] == comp_name].copy()
        subset = subset.sort_values("metric")
        y = np.arange(len(subset))
        offsets = 0.15 if "Cancer" in comp_name else -0.15
        ax1.scatter(subset["cohens_d"], y + offsets, color=color,
                    marker=marker, s=80, zorder=3, label=comp_name)
        for _, row in subset.iterrows():
            yi = np.where(subset["metric"].values == row["metric"])[0][0]
            star = "*" if row["significant"] else ""
            ax1.annotate(f'd={row["cohens_d"]:+.2f}{star}',
                         (row["cohens_d"], yi + offsets),
                         fontsize=7, ha="left", va="center",
                         xytext=(5, 0), textcoords="offset points")

    ax1.axvline(0, color="grey", linestyle="--", alpha=0.5)
    ax1.axvline(0.2, color="grey", linestyle=":", alpha=0.3)
    ax1.axvline(-0.2, color="grey", linestyle=":", alpha=0.3)
    ax1.axvline(0.8, color="grey", linestyle=":", alpha=0.3)
    ax1.axvline(-0.8, color="grey", linestyle=":", alpha=0.3)

    tea_metrics_sorted = sorted(tea_comp["metric"].unique())
    ax1.set_yticks(np.arange(len(tea_metrics_sorted)))
    ax1.set_yticklabels(tea_metrics_sorted, fontsize=10)
    ax1.set_xlabel("Cohen's d (OCD higher →)", fontsize=12)
    ax1.set_title("TEA Network Metrics", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # RQA effects
    for comp_name, color, marker in [
        ("OCD vs Cancer", "#E64B35", "o"),
        ("OCD vs Dep/PTSD", "#00A087", "s"),
    ]:
        subset = rqa_comp[rqa_comp["comparison"] == comp_name].copy()
        subset = subset.sort_values("metric")
        y = np.arange(len(subset))
        offsets = 0.15 if "Cancer" in comp_name else -0.15
        ax2.scatter(subset["cohens_d"], y + offsets, color=color,
                    marker=marker, s=80, zorder=3, label=comp_name)
        for _, row in subset.iterrows():
            yi = np.where(subset["metric"].values == row["metric"])[0][0]
            star = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["significant"] else ""
            ax2.annotate(f'd={row["cohens_d"]:+.2f}{star}',
                         (row["cohens_d"], yi + offsets),
                         fontsize=7, ha="left", va="center",
                         xytext=(5, 0), textcoords="offset points")

    ax2.axvline(0, color="grey", linestyle="--", alpha=0.5)
    ax2.axvline(0.2, color="grey", linestyle=":", alpha=0.3)
    ax2.axvline(-0.2, color="grey", linestyle=":", alpha=0.3)
    ax2.axvline(0.8, color="grey", linestyle=":", alpha=0.3)
    ax2.axvline(-0.8, color="grey", linestyle=":", alpha=0.3)

    rqa_metrics_sorted = sorted(rqa_comp["metric"].unique())
    ax2.set_yticks(np.arange(len(rqa_metrics_sorted)))
    ax2.set_yticklabels(rqa_metrics_sorted, fontsize=10)
    ax2.set_xlabel("Cohen's d (OCD higher →)", fontsize=12)
    ax2.set_title("RQA Metrics", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Effect Sizes: OCD vs Controls (length-matched)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, "effect_sizes_forest.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Figure 5: Recurrence Matrix Heatmaps ─────────────────────────────────────

def text_to_sentence_embeddings(text, _nlp):
    if len(text) > 45000:
        text = text[:45000]
    doc = _nlp(text)
    vecs = []
    for sent in doc.sents:
        if sent.vector_norm > 0 and len(sent.text.split()) >= 3:
            vecs.append(sent.vector / sent.vector_norm)
    return np.array(vecs) if vecs else None


def fig5_recurrence_plots():
    print("\n=== Figure 5: Recurrence Matrix Heatmaps ===")
    _nlp = load_nlp()

    samples = {
        "OCD": pick_sample(
            os.path.join(HERE, "metadata.csv"),
            os.path.join(HERE, "texts"),
            target_wc=1200, n=1
        ),
        "Cancer": pick_sample(
            os.path.join(HERE, "control_patient_stories", "metadata.csv"),
            os.path.join(HERE, "control_patient_stories", "texts"),
            target_wc=1200, n=1
        ),
        "Dep/PTSD": pick_sample(
            os.path.join(HERE, "control_mental_health", "metadata.csv"),
            os.path.join(HERE, "control_mental_health", "texts"),
            target_wc=1200, n=1
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    threshold = 0.92

    for idx, (group, items) in enumerate(samples.items()):
        ax = axes[idx]
        if not items:
            ax.set_title(f"{group}: no sample")
            continue
        slug, text, meta = items[0]
        wc = len(text.split())

        vecs = text_to_sentence_embeddings(text, _nlp)
        if vecs is None or len(vecs) < 5:
            ax.set_title(f"{group}: too few sentences")
            continue

        sim = vecs @ vecs.T
        R = (sim >= threshold).astype(np.float32)
        np.fill_diagonal(R, 0)

        n = R.shape[0]
        rr = R.sum() / (n * (n - 1)) if n > 1 else 0

        cmap = plt.cm.colors.ListedColormap(["white", PALETTE[group]])
        ax.imshow(R, cmap=cmap, origin="lower", aspect="equal", interpolation="none")
        ax.set_title(f"{group}\n({n} sent, RR={rr:.3f})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sentence index", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Sentence index", fontsize=11)

    plt.suptitle("Recurrence Plots (threshold=0.92)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, "recurrence_plots.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Figure 6: Violin plots for key metrics ───────────────────────────────────

def fig6_violin_plots():
    print("\n=== Figure 6: Violin Plots for Key Metrics ===")

    ocd_tea = pd.read_csv(os.path.join(HERE, "results", "tea_metrics.csv"))
    cancer_tea = pd.read_csv(os.path.join(HERE, "control_patient_stories", "results", "tea_metrics.csv"))
    mh_tea = pd.read_csv(os.path.join(HERE, "control_mental_health", "results", "tea_metrics.csv"))

    ocd_rqa = pd.read_csv(os.path.join(HERE, "results", "rqa_ocd.csv"))
    cancer_rqa = pd.read_csv(os.path.join(HERE, "results", "rqa_cancer.csv"))
    mh_rqa = pd.read_csv(os.path.join(HERE, "results", "rqa_mh.csv"))

    metrics = [
        ("first_person_ratio", "First-Person Agent Ratio", "tea"),
        ("RR", "Recurrence Rate", "rqa"),
        ("DET", "Determinism", "rqa"),
        ("ENTR", "Recurrence Entropy", "rqa"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for idx, (col, label, source) in enumerate(metrics):
        ax = axes[idx]
        if source == "tea":
            datasets = [ocd_tea[col].dropna(), cancer_tea[col].dropna(), mh_tea[col].dropna()]
        else:
            datasets = [ocd_rqa[col].dropna(), cancer_rqa[col].dropna(), mh_rqa[col].dropna()]

        parts = ax.violinplot(datasets, positions=[0, 1, 2], showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            color = list(PALETTE.values())[i]
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["OCD", "Cancer", "Dep/PTSD"], fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Distribution of Key Metrics Across Groups",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, "violin_plots.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Figure 7: Top Agents by Degree Centrality ────────────────────────────────

def fig7_top_agents():
    print("\n=== Figure 7: Top Agents Degree Centrality ===")

    groups = {
        "OCD": pick_sample(
            os.path.join(HERE, "metadata.csv"),
            os.path.join(HERE, "texts"),
            target_wc=1500, n=3
        ),
        "Cancer": pick_sample(
            os.path.join(HERE, "control_patient_stories", "metadata.csv"),
            os.path.join(HERE, "control_patient_stories", "texts"),
            target_wc=1500, n=3
        ),
        "Dep/PTSD": pick_sample(
            os.path.join(HERE, "control_mental_health", "metadata.csv"),
            os.path.join(HERE, "control_mental_health", "texts"),
            target_wc=1500, n=3
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (group, items) in enumerate(groups.items()):
        ax = axes[idx]
        if not items:
            ax.set_title(f"{group}: no data")
            continue

        all_dfs = []
        for slug, text, meta in items:
            df = extract_tea(text)
            if df is not None and len(df) > 0:
                all_dfs.append(df)

        if not all_dfs:
            ax.set_title(f"{group}: extraction failed")
            continue

        merged = pd.concat(all_dfs, ignore_index=True)
        try:
            centrality = tea_weighted_degree_centrality(merged, "Agent")
        except Exception as e:
            print(f"    {group} centrality error: {e}")
            ax.set_title(f"{group}: error")
            continue

        top = centrality.head(10)
        color = PALETTE[group]
        y = np.arange(len(top))
        ax.barh(y, top["degree_centrality"], color=color, alpha=0.8, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(top["node"].values, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Weighted Degree Centrality", fontsize=11)
        ax.set_title(f"{group}\nTop Agents", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Agent Centrality (Stella's tea_weighted_degree_centrality)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, "top_agents_centrality.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Figure 8: Integrated Summary Panel ───────────────────────────────────────

def fig8_summary_panel():
    print("\n=== Figure 8: Integrated Summary Panel ===")

    tea_comp = pd.read_csv(os.path.join(HERE, "results", "three_group_comparison.csv"))
    rqa_comp = pd.read_csv(os.path.join(HERE, "results", "rqa_three_group_comparison.csv"))

    key_findings = []
    for _, row in tea_comp.iterrows():
        if row["significant"]:
            key_findings.append(("TEA", row["comparison"], row["metric"],
                                 row["cohens_d"], row["p_value"], row["effect"]))
    for _, row in rqa_comp.iterrows():
        if row["significant"]:
            key_findings.append(("RQA", row["comparison"], row["metric"],
                                 row["cohens_d"], row["p_value"], row["effect"]))

    key_findings.sort(key=lambda x: -abs(x[3]))

    fig, ax = plt.subplots(figsize=(14, max(6, len(key_findings) * 0.4 + 2)))
    ax.axis("off")

    headers = ["Method", "Comparison", "Metric", "Cohen's d", "p-value", "Effect"]
    col_x = [0.02, 0.10, 0.25, 0.55, 0.70, 0.85]

    y = 0.95
    for i, header in enumerate(headers):
        ax.text(col_x[i], y, header, fontsize=11, fontweight="bold",
                transform=ax.transAxes, va="top")
    y -= 0.04
    ax.plot([0.01, 0.99], [y, y], color="black", linewidth=1,
            transform=ax.transAxes, clip_on=False)
    y -= 0.02

    for method, comp, metric, d, p, eff in key_findings:
        color = "#E64B35" if "Cancer" in comp else "#00A087"
        direction = "OCD higher" if d > 0 else "OCD lower"
        for i, val in enumerate([method, comp, metric,
                                 f"{d:+.3f} ({direction})",
                                 f"{p:.6f}", eff]):
            fw = "bold" if i == 5 and eff == "LARGE" else "normal"
            ax.text(col_x[i], y, val, fontsize=9, fontweight=fw,
                    color=color if i == 1 else "black",
                    transform=ax.transAxes, va="top")
        y -= 0.035

    ax.set_title("Significant Findings: OCD Cognitive Network Analysis",
                 fontsize=15, fontweight="bold", pad=20)

    plt.tight_layout()
    fname = os.path.join(FIG_DIR, "summary_table.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Generating figures with Massimo Stella's TEA_Networks functions")
    print("=" * 70)

    # Figures that don't need spaCy
    fig2_tea_comparison()
    fig3_rqa_comparison()
    fig4_effect_sizes()
    fig8_summary_panel()

    # Figures that need spaCy
    fig1_tea_networks()
    fig5_recurrence_plots()
    fig6_violin_plots()
    fig7_top_agents()

    print("\n" + "=" * 70)
    figs = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
    print(f"  Done! Generated {len(figs)} figures in {FIG_DIR}/")
    for f in sorted(figs):
        size_kb = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
        print(f"    {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
