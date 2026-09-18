"""
Publication-quality figures and formal baseline test for TEA Multiplex paper.

Figures:
  1. Heatmap of layer dominance × group
  2. Radar chart of first-person emotional profiles
  3. Forest plot of BH-surviving effect sizes (Multiplex only)
  4. Added-value: logistic regression AUC comparison (word counts vs multiplex)
"""
import os
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from nrclex import NRCLex

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

PLUTCHIK = ["fear", "anger", "sadness", "joy", "trust", "disgust", "surprise", "anticipation"]
PLUTCHIK_COLORS = {
    "fear":         "#2ca02c",
    "anger":        "#d62728",
    "sadness":      "#1f77b4",
    "joy":          "#ffdd57",
    "trust":        "#8cc63f",
    "disgust":      "#9467bd",
    "surprise":     "#17becf",
    "anticipation": "#ff7f0e",
}

GROUP_COLORS = {
    "OCD":      "#d62728",
    "Cancer":   "#1f77b4",
    "Dep/PTSD": "#2ca02c",
}

_nrc = NRCLex("init")
NRC_LEX = _nrc.__lexicon__


def load_group(fname):
    path = os.path.join(HERE, "results", fname)
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in row:
                if k not in ("slug", "dominant_emotion"):
                    try:
                        row[k] = float(row[k]) if row[k] not in ("", "None") else np.nan
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def load_bh():
    path = os.path.join(HERE, "results", "all_comparisons_bh_corrected.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["cohens_d"] = float(row["cohens_d"])
            row["p_value"] = float(row["p_value"])
            row["p_adj_bh"] = float(row["p_adj_bh"])
            row["sig_bh"] = row["sig_bh"] == "True"
            rows.append(row)
    return rows


# ============================================================================
# FIGURE 1: Heatmap — Layer Dominance × Group
# ============================================================================

def fig_heatmap(ocd, cancer, mh):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    data = np.zeros((3, 8))
    for gi, (group, label) in enumerate([(ocd, "OCD"), (cancer, "Cancer"), (mh, "Dep/PTSD")]):
        for ei, emo in enumerate(PLUTCHIK):
            vals = [r[f"dom_{emo}"] for r in group if not np.isnan(r.get(f"dom_{emo}", np.nan))]
            data[gi, ei] = np.mean(vals)

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.05, vmax=0.22)

    ax.set_xticks(range(8))
    ax.set_xticklabels([e.capitalize() for e in PLUTCHIK], fontsize=11, rotation=35, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["OCD", "Cancer", "Dep/PTSD"], fontsize=12)

    for i in range(3):
        for j in range(8):
            val = data[i, j]
            color = "white" if val > 0.16 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Layer Dominance", fontsize=11)

    ax.set_title("Emotional Layer Dominance Across Clinical Groups", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig1_heatmap_dominance.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 2: Radar Chart — First-Person Emotional Profile
# ============================================================================

def fig_radar(ocd, cancer, mh):
    angles = np.linspace(0, 2 * np.pi, len(PLUTCHIK), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for group, label, color in [(ocd, "OCD", GROUP_COLORS["OCD"]),
                                 (cancer, "Cancer", GROUP_COLORS["Cancer"]),
                                 (mh, "Dep/PTSD", GROUP_COLORS["Dep/PTSD"])]:
        values = []
        errors = []
        for emo in PLUTCHIK:
            vals = [r[f"fp_{emo}"] for r in group if not np.isnan(r.get(f"fp_{emo}", np.nan))]
            values.append(np.mean(vals))
            errors.append(np.std(vals) / np.sqrt(len(vals)))
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2.2, label=label, color=color, markersize=5)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([e.capitalize() for e in PLUTCHIK], fontsize=11)
    ax.set_ylim(0, 0.12)
    ax.set_yticks([0.02, 0.04, 0.06, 0.08, 0.10])
    ax.set_yticklabels(["0.02", "0.04", "0.06", "0.08", "0.10"], fontsize=8, color="gray")

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=11, framealpha=0.9)
    ax.set_title("First-Person Emotional Profile\n(Agent = I/me/my/we)", fontsize=13, fontweight="bold", y=1.08)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig2_radar_first_person.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 3: Forest Plot — BH-Surviving Effect Sizes (Multiplex only)
# ============================================================================

def fig_forest(bh_results):
    multiplex_sig = [r for r in bh_results if r["sig_bh"] and r["analysis"] == "Multiplex"]
    multiplex_sig.sort(key=lambda x: x["cohens_d"])

    if not multiplex_sig:
        print("  No significant multiplex results for forest plot")
        return

    n = len(multiplex_sig)
    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.35)))

    y_positions = range(n)

    for i, r in enumerate(multiplex_sig):
        d = r["cohens_d"]
        comparison = r["comparison"].replace("OCD vs ", "vs ")
        metric = r["metric"]
        label = f"{metric} ({comparison})"

        color = GROUP_COLORS["OCD"] if "Cancer" in r["comparison"] else GROUP_COLORS["Dep/PTSD"]

        ax.barh(i, d, height=0.6, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.text(d + (0.03 if d >= 0 else -0.03), i,
                f"d={d:+.2f}",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=8, color="black")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{r['metric']} ({r['comparison'].replace('OCD vs ', 'vs ')})"
                        for r in multiplex_sig], fontsize=9)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(x=0.2, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(x=-0.2, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(x=0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(x=-0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(x=0.8, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(x=-0.8, color="gray", linewidth=0.5, linestyle=":")

    ax.set_xlabel("Cohen's d (OCD higher →  |  ← Control higher)", fontsize=11)
    ax.set_title("TEA Multiplex: Effect Sizes Surviving BH Correction", fontsize=13, fontweight="bold")

    legend_elements = [
        mpatches.Patch(color=GROUP_COLORS["OCD"], alpha=0.75, label="OCD vs Cancer"),
        mpatches.Patch(color=GROUP_COLORS["Dep/PTSD"], alpha=0.75, label="OCD vs Dep/PTSD"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig3_forest_multiplex.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 4: Pipeline schematic
# ============================================================================

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.5, 1.5, 2.2, 1.5, "Narrative\nText", "#e8e8e8"),
        (3.2, 1.5, 2.2, 1.5, "spaCy NLP\n+ TEA SVO\nExtraction", "#d4e6f1"),
        (5.9, 1.5, 2.2, 1.5, "NRC EmoLex\nEmotion\nClassification", "#fadbd8"),
        (8.6, 1.5, 2.8, 1.5, "Multiplex\nNetwork\n(9 layers)", "#d5f5e3"),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor="#333333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#333333")

    for x1, x2 in [(2.7, 3.2), (5.4, 5.9), (8.1, 8.6)]:
        ax.annotate("", xy=(x2, 2.25), xytext=(x1, 2.25),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=2))

    ax.text(1.6, 0.7, '"I fear losing\ncontrol again"', ha="center", fontsize=9,
            style="italic", color="#666666")
    ax.text(4.3, 0.7, "I → fear → control\n(Agent→Event→Target)", ha="center", fontsize=9,
            color="#666666", family="monospace")
    ax.text(7.0, 0.7, "fear ✓  anticipation ✓\nanger ✗  joy ✗ ...", ha="center", fontsize=9,
            color="#666666")
    ax.text(10.0, 0.7, "Syntactic + 8 Plutchik\nemotional layers", ha="center", fontsize=9,
            color="#666666")

    ax.set_title("TEA Multiplex Pipeline", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig4_pipeline.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============================================================================
# FIGURE 5 + TABLE: Logistic Regression AUC — Word Counts vs Multiplex
# ============================================================================

def compute_emotion_word_counts(text):
    words = text.lower().split()
    n = len(words)
    if n == 0:
        return None
    counts = {e: 0 for e in PLUTCHIK}
    for w in words:
        emos = NRC_LEX.get(w, [])
        for e in emos:
            if e in counts:
                counts[e] += 1
    return {f"wc_{e}": counts[e] / n for e in PLUTCHIK}


def run_baseline_test(ocd, cancer, mh):
    print("\n" + "=" * 90)
    print("  BASELINE TEST: Logistic Regression AUC — Word Counts vs Multiplex Metrics")
    print("=" * 90)

    corpora_dirs = {
        "OCD": os.path.join(HERE, "texts"),
        "Cancer": os.path.join(HERE, "control_patient_stories", "texts"),
        "Dep/PTSD": os.path.join(HERE, "control_mental_health", "texts"),
    }

    groups_data = {"OCD": ocd, "Cancer": cancer, "Dep/PTSD": mh}

    for group_name, data in groups_data.items():
        texts_dir = corpora_dirs[group_name]
        for row in data:
            slug = row["slug"]
            txt_path = os.path.join(texts_dir, f"{slug}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, encoding="utf-8") as f:
                    text = f.read()
                wc = compute_emotion_word_counts(text)
                if wc:
                    row.update(wc)

    wc_features = [f"wc_{e}" for e in PLUTCHIK]
    multiplex_features = (
        [f"dom_{e}" for e in PLUTCHIK]
        + ["participation_coeff", "emotional_concentration"]
        + [f"fp_{e}" for e in PLUTCHIK]
        + [f"coverage_{e}" for e in PLUTCHIK]
    )

    comparisons = [
        ("OCD", "Cancer", ocd, cancer),
        ("OCD", "Dep/PTSD", ocd, mh),
    ]

    auc_results = []

    for name_a, name_b, data_a, data_b in comparisons:
        print(f"\n  {name_a} vs {name_b}:")

        valid_a = [r for r in data_a if all(f"wc_{e}" in r for e in PLUTCHIK)]
        valid_b = [r for r in data_b if all(f"wc_{e}" in r for e in PLUTCHIK)]

        all_data = valid_a + valid_b
        labels = np.array([1] * len(valid_a) + [0] * len(valid_b))

        for model_name, features in [
            ("Word Counts Only", wc_features),
            ("Multiplex Only", multiplex_features),
            ("Word Counts + Multiplex", wc_features + multiplex_features),
        ]:
            X = []
            valid_idx = []
            for i, row in enumerate(all_data):
                feat_vals = []
                skip = False
                for f in features:
                    v = row.get(f)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        skip = True
                        break
                    feat_vals.append(float(v))
                if not skip:
                    X.append(feat_vals)
                    valid_idx.append(i)

            X = np.array(X)
            y = labels[valid_idx]

            if len(X) < 30 or len(np.unique(y)) < 2:
                print(f"    {model_name}: insufficient data")
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = []
            for train_idx, test_idx in cv.split(X_scaled, y):
                clf = LogisticRegression(max_iter=1000, C=1.0, penalty="l2", random_state=42)
                clf.fit(X_scaled[train_idx], y[train_idx])
                probs = clf.predict_proba(X_scaled[test_idx])[:, 1]
                if len(np.unique(y[test_idx])) > 1:
                    aucs.append(roc_auc_score(y[test_idx], probs))

            mean_auc = np.mean(aucs) if aucs else 0
            std_auc = np.std(aucs) if aucs else 0

            print(f"    {model_name:30s}: AUC = {mean_auc:.3f} ± {std_auc:.3f}  (n={len(X)}, {len(features)} features)")
            auc_results.append({
                "comparison": f"{name_a} vs {name_b}",
                "model": model_name,
                "auc_mean": round(mean_auc, 3),
                "auc_std": round(std_auc, 3),
                "n": len(X),
                "n_features": len(features),
            })

    # Save
    path = os.path.join(HERE, "results", "baseline_auc_comparison.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(auc_results[0].keys()))
        w.writeheader()
        w.writerows(auc_results)
    print(f"\n  Saved {path}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax_i, (name_a, name_b) in enumerate([("OCD", "Cancer"), ("OCD", "Dep/PTSD")]):
        ax = axes[ax_i]
        comp_label = f"{name_a} vs {name_b}"
        subset = [r for r in auc_results if r["comparison"] == comp_label]

        models = [r["model"] for r in subset]
        aucs_mean = [r["auc_mean"] for r in subset]
        aucs_std = [r["auc_std"] for r in subset]

        colors = ["#a9cce3", "#f5b7b1", "#abebc6"]
        bars = ax.barh(range(len(models)), aucs_mean, xerr=aucs_std,
                       height=0.5, color=colors[:len(models)], edgecolor="gray",
                       capsize=4, alpha=0.85)

        for i, (m, s) in enumerate(zip(aucs_mean, aucs_std)):
            ax.text(m + s + 0.02, i, f"{m:.3f}", va="center", fontsize=10, fontweight="bold")

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("AUC (5-fold CV)", fontsize=11)
        ax.set_title(comp_label, fontsize=12, fontweight="bold")
        ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=0.8)

    fig.suptitle("Classification Performance: Word Counts vs Multiplex Metrics",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig5_auc_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    return auc_results


# ============================================================================
# FIGURE 6: Coverage bar chart
# ============================================================================

def fig_coverage(ocd, cancer, mh):
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(PLUTCHIK))
    width = 0.25

    for offset, (group, label, color) in enumerate([
        (ocd, "OCD", GROUP_COLORS["OCD"]),
        (cancer, "Cancer", GROUP_COLORS["Cancer"]),
        (mh, "Dep/PTSD", GROUP_COLORS["Dep/PTSD"]),
    ]):
        means = []
        sems = []
        for emo in PLUTCHIK:
            vals = [r[f"coverage_{emo}"] for r in group if not np.isnan(r.get(f"coverage_{emo}", np.nan))]
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))

        ax.bar(x + (offset - 1) * width, means, width, yerr=sems,
               label=label, color=color, alpha=0.8, capsize=3, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in PLUTCHIK], fontsize=11, rotation=30, ha="right")
    ax.set_ylabel("Coverage (fraction of syntactic nodes)", fontsize=11)
    ax.set_title("Emotional Layer Coverage Across Clinical Groups", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 0.40)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig6_coverage.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def main():
    print("=" * 90)
    print("  PUBLICATION FIGURES")
    print("=" * 90)

    ocd = load_group("multiplex_ocd.csv")
    cancer = load_group("multiplex_cancer.csv")
    mh = load_group("multiplex_mh.csv")
    bh = load_bh()

    print(f"\n  Loaded: OCD={len(ocd)}, Cancer={len(cancer)}, Dep/PTSD={len(mh)}, BH results={len(bh)}")

    print("\n  Generating figures...")
    fig_heatmap(ocd, cancer, mh)
    fig_radar(ocd, cancer, mh)
    fig_forest(bh)
    fig_pipeline()
    fig_coverage(ocd, cancer, mh)

    auc_results = run_baseline_test(ocd, cancer, mh)

    print("\n" + "=" * 90)
    print("  ALL FIGURES SAVED TO:", OUTDIR)
    print("=" * 90)


if __name__ == "__main__":
    main()
