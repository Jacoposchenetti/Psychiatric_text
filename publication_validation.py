"""
Publication validation:
  A) Benjamini-Hochberg correction on all comparisons (TEA, RQA, Multiplex)
  B) Null models: configuration model for TEA networks, shuffled sentences for RQA
  C) Added-value test: does multiplex structure add info beyond emotion word counts?
"""
import sys
import os
import csv
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import spacy
from scipy import stats
from nrclex import NRCLex

sys.path.insert(0, r"C:\Users\jsche\Desktop\NLP\TEA_Networks")

HERE = os.path.dirname(__file__)
random.seed(42)
np.random.seed(42)

PLUTCHIK = ["fear", "anger", "sadness", "joy", "trust", "disgust", "surprise", "anticipation"]
_nrc = NRCLex("init")
NRC_LEX = _nrc.__lexicon__


# ============================================================================
# PART A: Benjamini-Hochberg correction
# ============================================================================

def benjamini_hochberg(p_values, alpha=0.05):
    """Apply BH correction. Returns adjusted p-values and boolean mask."""
    n = len(p_values)
    if n == 0:
        return [], []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    significant = [False] * n

    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adj_p = min(prev_adj, p * n / rank)
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        significant[orig_idx] = adj_p < alpha
        prev_adj = adj_p

    return adjusted, significant


def correct_comparisons():
    """Apply BH correction to all three sets of results."""
    files = {
        "TEA": ("three_group_comparison.csv", "ocd_mean", "ctrl_mean"),
        "RQA": ("rqa_three_group_comparison.csv", "a_mean", "b_mean"),
        "Multiplex": ("multiplex_comparison.csv", "a_mean", "b_mean"),
    }

    all_results = []

    for analysis, (fname, col_a, col_b) in files.items():
        path = os.path.join(HERE, "results", fname)
        if not os.path.exists(path):
            print(f"  Skipping {analysis}: {fname} not found")
            continue

        rows = []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["analysis"] = analysis
                row["p_value"] = float(row["p_value"])
                row["cohens_d"] = float(row["cohens_d"])
                rows.append(row)

        all_results.extend(rows)

    if not all_results:
        return []

    p_values = [r["p_value"] for r in all_results]
    adj_p, sig_bh = benjamini_hochberg(p_values)

    for i, r in enumerate(all_results):
        r["p_adj_bh"] = round(adj_p[i], 6)
        r["sig_bh"] = sig_bh[i]
        r["sig_original"] = r.get("significant", "True") == "True" or r.get("significant", True) is True

    return all_results


def print_bh_results(results):
    """Print summary of BH correction impact."""
    print("\n" + "=" * 110)
    print("  PART A: BENJAMINI-HOCHBERG CORRECTION (all analyses combined)")
    print("=" * 110)

    n_total = len(results)
    n_orig_sig = sum(1 for r in results if r["sig_original"])
    n_bh_sig = sum(1 for r in results if r["sig_bh"])
    n_lost = n_orig_sig - n_bh_sig

    print(f"\n  Total tests: {n_total}")
    print(f"  Originally significant (p<0.05): {n_orig_sig}")
    print(f"  Surviving BH correction: {n_bh_sig}")
    print(f"  Lost after correction: {n_lost}")

    print(f"\n  {'Analysis':10s} {'Comparison':30s} {'Metric':40s} {'d':>7s} {'p_raw':>10s} {'p_adj':>10s} {'BH':>5s}")
    print("  " + "-" * 115)

    for r in sorted(results, key=lambda x: (-int(x["sig_bh"]), -abs(x["cohens_d"]))):
        if not r["sig_original"] and not r["sig_bh"]:
            continue

        status = "YES" if r["sig_bh"] else "lost"
        comparison = r.get("comparison", "")
        metric = r.get("metric", "")
        d = r["cohens_d"]

        print(f"  {r['analysis']:10s} {comparison:30s} {metric:40s} "
              f"{d:+7.3f} {r['p_value']:10.6f} {r['p_adj_bh']:10.6f} {status:>5s}")

    # Summary by analysis
    print("\n  Summary by analysis:")
    for analysis in ["TEA", "RQA", "Multiplex"]:
        subset = [r for r in results if r["analysis"] == analysis]
        orig = sum(1 for r in subset if r["sig_original"])
        surv = sum(1 for r in subset if r["sig_bh"])
        print(f"    {analysis:10s}: {orig} originally sig -> {surv} surviving BH ({orig-surv} lost)")

    return results


# ============================================================================
# PART B: Null Models
# ============================================================================

def load_nlp():
    print("  Loading en_core_web_lg...")
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    return nlp


def build_tea_graph_from_text(text, nlp):
    """Build a TEA DiGraph from text."""
    from teanets.svo_extraction import extract_svos
    if len(text) > 49000:
        text = text[:49000]
    doc = nlp(text)
    try:
        df = extract_svos(doc)
        if df is None or len(df) == 0:
            return None
        G = nx.DiGraph()
        for _, row in df.iterrows():
            n1 = str(row["Node 1"]).strip().lower()
            n2 = str(row["Node 2"]).strip().lower()
            if not n1 or not n2 or n1 == "nan" or n2 == "nan":
                continue
            if G.has_edge(n1, n2):
                G[n1][n2]["weight"] += 1
            else:
                G.add_edge(n1, n2, weight=1)
        return G if G.number_of_nodes() > 5 else None
    except Exception:
        return None


def configuration_model_metrics(G, n_rewires=100):
    """
    Generate configuration model null: random graphs with same degree sequence.
    Returns mean and std of key metrics across n_rewires realizations.
    """
    if G.number_of_nodes() < 5:
        return None

    in_seq = [d for _, d in G.in_degree()]
    out_seq = [d for _, d in G.out_degree()]

    null_metrics = {"density": [], "clustering": [], "lscc_frac": []}

    for _ in range(n_rewires):
        try:
            G_rand = nx.directed_configuration_model(in_seq, out_seq, seed=random.randint(0, 99999))
            G_rand = nx.DiGraph(G_rand)
            G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

            n = G_rand.number_of_nodes()
            null_metrics["density"].append(nx.density(G_rand))

            Gu = G_rand.to_undirected()
            null_metrics["clustering"].append(nx.average_clustering(Gu) if n > 2 else 0)

            sccs = list(nx.strongly_connected_components(G_rand))
            lscc = max(len(c) for c in sccs) if sccs else 0
            null_metrics["lscc_frac"].append(lscc / n if n > 0 else 0)
        except Exception:
            continue

    if not null_metrics["density"]:
        return None

    return {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in null_metrics.items()}


def compute_real_metrics(G):
    """Compute real network metrics for comparison with null."""
    n = G.number_of_nodes()
    if n < 5:
        return None

    Gu = G.to_undirected()
    sccs = list(nx.strongly_connected_components(G))
    lscc = max(len(c) for c in sccs) if sccs else 0

    return {
        "density": nx.density(G),
        "clustering": nx.average_clustering(Gu) if n > 2 else 0,
        "lscc_frac": lscc / n,
    }


def run_null_model_tea(nlp, n_sample=30):
    """Test TEA network metrics against configuration model null."""
    print("\n" + "=" * 110)
    print("  PART B1: NULL MODEL — Configuration Model for TEA Networks")
    print("=" * 110)

    corpora = {
        "OCD": (os.path.join(HERE, "texts"), os.path.join(HERE, "metadata.csv")),
        "Cancer": (os.path.join(HERE, "control_patient_stories", "texts"),
                   os.path.join(HERE, "control_patient_stories", "metadata.csv")),
        "Dep/PTSD": (os.path.join(HERE, "control_mental_health", "texts"),
                     os.path.join(HERE, "control_mental_health", "metadata.csv")),
    }

    all_z = {g: {"density": [], "clustering": [], "lscc_frac": []} for g in corpora}

    for group, (texts_dir, meta_path) in corpora.items():
        meta = []
        with open(meta_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta.append(row)

        if "word_count" in meta[0]:
            meta = [r for r in meta if int(r.get("word_count", 0)) >= 300]

        sample = random.sample(meta, min(n_sample, len(meta)))

        print(f"\n  {group}: testing {len(sample)} texts against configuration model (100 rewires each)")

        for i, row in enumerate(sample):
            slug = row["slug"]
            txt_path = os.path.join(texts_dir, f"{slug}.txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path, encoding="utf-8") as f:
                text = f.read()

            G = build_tea_graph_from_text(text, nlp)
            if G is None:
                continue

            real = compute_real_metrics(G)
            null = configuration_model_metrics(G, n_rewires=100)
            if real is None or null is None:
                continue

            for metric in ["density", "clustering", "lscc_frac"]:
                if null[metric]["std"] > 0:
                    z = (real[metric] - null[metric]["mean"]) / null[metric]["std"]
                else:
                    z = 0.0
                all_z[group][metric].append(z)

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(sample)}] done")

    # Report
    print(f"\n  {'Group':15s} {'Metric':15s} {'Mean z':>10s} {'p (z!=0)':>12s} {'Interpretation':>20s}")
    print("  " + "-" * 80)

    null_results = []
    for group in corpora:
        for metric in ["density", "clustering", "lscc_frac"]:
            zs = all_z[group][metric]
            if len(zs) < 5:
                continue
            mean_z = np.mean(zs)
            t_stat, p_val = stats.ttest_1samp(zs, 0)
            interpretation = "DIFFERS from null" if p_val < 0.05 else "~same as null"
            print(f"  {group:15s} {metric:15s} {mean_z:+10.3f} {p_val:12.6f} {interpretation:>20s}")
            null_results.append({
                "group": group, "metric": metric,
                "mean_z": round(mean_z, 3), "p_value": round(p_val, 6),
                "n": len(zs), "differs_from_null": p_val < 0.05,
            })

    return null_results


def run_null_model_rqa(nlp, n_sample=30):
    """Test RQA metrics against shuffled-sentences null."""
    print("\n" + "=" * 110)
    print("  PART B2: NULL MODEL — Shuffled Sentences for RQA")
    print("=" * 110)

    corpora = {
        "OCD": (os.path.join(HERE, "texts"), os.path.join(HERE, "metadata.csv")),
        "Cancer": (os.path.join(HERE, "control_patient_stories", "texts"),
                   os.path.join(HERE, "control_patient_stories", "metadata.csv")),
        "Dep/PTSD": (os.path.join(HERE, "control_mental_health", "texts"),
                     os.path.join(HERE, "control_mental_health", "metadata.csv")),
    }

    def text_to_embeddings(text):
        if len(text) > 45000:
            text = text[:45000]
        doc = nlp(text)
        vecs = []
        for sent in doc.sents:
            if sent.vector_norm > 0 and len(sent.text.split()) >= 3:
                vecs.append(sent.vector / sent.vector_norm)
        return np.array(vecs) if vecs else None

    def compute_rr(vecs, threshold):
        sim = vecs @ vecs.T
        np.fill_diagonal(sim, 0)
        R = (sim >= threshold).astype(np.int8)
        n = R.shape[0]
        total = n * (n - 1)
        return R.sum() / total if total > 0 else 0

    def compute_det(vecs, threshold):
        sim = vecs @ vecs.T
        np.fill_diagonal(sim, 0)
        R = (sim >= threshold).astype(np.int8)
        n = R.shape[0]
        total_recurrent = R.sum()
        if total_recurrent == 0:
            return 0

        diag_lengths = []
        for k in list(range(1, n)) + list(range(-n + 1, 0)):
            diag = np.diag(R, k)
            length = 0
            for val in diag:
                if val:
                    length += 1
                else:
                    if length >= 2:
                        diag_lengths.append(length)
                    length = 0
            if length >= 2:
                diag_lengths.append(length)

        points_on_diag = sum(diag_lengths) if diag_lengths else 0
        return points_on_diag / total_recurrent

    threshold = 0.9232  # same as three-group RQA

    all_deltas = {g: {"rr": [], "det": []} for g in corpora}

    for group, (texts_dir, meta_path) in corpora.items():
        meta = []
        with open(meta_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta.append(row)

        if "word_count" in meta[0]:
            meta = [r for r in meta if int(r.get("word_count", 0)) >= 300]

        sample = random.sample(meta, min(n_sample, len(meta)))

        print(f"\n  {group}: testing {len(sample)} texts (real vs 20 sentence-shuffles each)")

        for i, row in enumerate(sample):
            slug = row["slug"]
            txt_path = os.path.join(texts_dir, f"{slug}.txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path, encoding="utf-8") as f:
                text = f.read()

            vecs = text_to_embeddings(text)
            if vecs is None or len(vecs) < 10:
                continue

            real_rr = compute_rr(vecs, threshold)
            real_det = compute_det(vecs, threshold)

            null_rrs, null_dets = [], []
            for _ in range(20):
                idx = np.random.permutation(len(vecs))
                shuffled = vecs[idx]
                null_rrs.append(compute_rr(shuffled, threshold))
                null_dets.append(compute_det(shuffled, threshold))

            # RR should be similar (shuffling doesn't change pairwise similarities)
            # DET should drop (diagonal structure depends on order)
            all_deltas[group]["rr"].append(real_rr - np.mean(null_rrs))
            all_deltas[group]["det"].append(real_det - np.mean(null_dets))

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(sample)}] done")

    print(f"\n  {'Group':15s} {'Metric':10s} {'Real-Null':>12s} {'p':>12s} {'Interpretation':>30s}")
    print("  " + "-" * 85)

    null_results = []
    for group in corpora:
        for metric in ["rr", "det"]:
            deltas = all_deltas[group][metric]
            if len(deltas) < 5:
                continue
            mean_delta = np.mean(deltas)
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            if metric == "rr":
                expected = "~0 (shuffling preserves pairwise sim)"
            else:
                expected = ">0 (real text has sequential structure)"
            print(f"  {group:15s} {metric.upper():10s} {mean_delta:+12.6f} {p_val:12.6f} {expected:>30s}")
            null_results.append({
                "group": group, "metric": metric.upper(),
                "mean_delta": round(mean_delta, 6), "p_value": round(p_val, 6),
                "n": len(deltas),
            })

    return null_results


# ============================================================================
# PART C: Added Value — Multiplex vs Simple Word Counts
# ============================================================================

def compute_emotion_word_counts(text):
    """Simple emotion word proportions (no network structure)."""
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


def run_added_value_test():
    """
    Test whether multiplex metrics add info beyond simple emotion word counts.
    For each group pair, check if multiplex metrics still differ after controlling
    for word-count-based emotion proportions.
    """
    print("\n" + "=" * 110)
    print("  PART C: ADDED VALUE — Multiplex structure vs simple word counts")
    print("=" * 110)

    # Load multiplex results
    groups = {}
    for name, fname in [("OCD", "multiplex_ocd.csv"), ("Cancer", "multiplex_cancer.csv"), ("Dep/PTSD", "multiplex_mh.csv")]:
        path = os.path.join(HERE, "results", fname)
        rows = []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        groups[name] = rows

    # For each text, compute simple word-count emotions
    corpora_dirs = {
        "OCD": os.path.join(HERE, "texts"),
        "Cancer": os.path.join(HERE, "control_patient_stories", "texts"),
        "Dep/PTSD": os.path.join(HERE, "control_mental_health", "texts"),
    }

    for group_name, rows in groups.items():
        texts_dir = corpora_dirs[group_name]
        for row in rows:
            slug = row["slug"]
            txt_path = os.path.join(texts_dir, f"{slug}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, encoding="utf-8") as f:
                    text = f.read()
                wc = compute_emotion_word_counts(text)
                if wc:
                    row.update(wc)

    # For key multiplex metrics, compute correlation with word-count equivalents
    print("\n  Correlation between multiplex metrics and simple word counts:")
    print(f"\n  {'Multiplex metric':35s} {'Word-count equiv':20s} {'Pearson r':>10s} {'p':>10s} {'Shared var':>12s}")
    print("  " + "-" * 95)

    multiplex_vs_wc = [
        ("dom_fear", "wc_fear", "Fear dominance"),
        ("dom_anger", "wc_anger", "Anger dominance"),
        ("dom_sadness", "wc_sadness", "Sadness dominance"),
        ("dom_joy", "wc_joy", "Joy dominance"),
        ("dom_trust", "wc_trust", "Trust dominance"),
        ("dom_disgust", "wc_disgust", "Disgust dominance"),
        ("dom_surprise", "wc_surprise", "Surprise dominance"),
        ("dom_anticipation", "wc_anticipation", "Anticipation dominance"),
    ]

    all_texts = []
    for rows in groups.values():
        all_texts.extend(rows)

    for mx_col, wc_col, label in multiplex_vs_wc:
        mx_vals = []
        wc_vals = []
        for row in all_texts:
            mx = row.get(mx_col)
            wc = row.get(wc_col)
            if mx is not None and wc is not None:
                try:
                    mx_vals.append(float(mx))
                    wc_vals.append(float(wc))
                except (ValueError, TypeError):
                    continue

        if len(mx_vals) < 20:
            continue

        r, p = stats.pearsonr(mx_vals, wc_vals)
        shared = r ** 2
        print(f"  {label:35s} {wc_col:20s} {r:+10.3f} {p:10.6f} {shared:11.1%}")

    # Key test: do structural metrics (participation, inter-layer correlation)
    # correlate with ANY word count metric?
    print("\n  Structural metrics vs any word-count metric:")
    print(f"  {'Structural metric':35s} {'Max |r| with any wc':>22s} {'Which wc':>20s}")
    print("  " + "-" * 80)

    structural = ["participation_coeff", "emotional_concentration"]
    for emo in PLUTCHIK:
        structural.append(f"corr_synt_{emo}")

    wc_cols = [f"wc_{e}" for e in PLUTCHIK]

    for s_col in structural:
        max_r = 0
        max_wc = ""
        for wc_col in wc_cols:
            s_vals, w_vals = [], []
            for row in all_texts:
                s = row.get(s_col)
                w = row.get(wc_col)
                if s is not None and w is not None:
                    try:
                        s_vals.append(float(s))
                        w_vals.append(float(w))
                    except (ValueError, TypeError):
                        continue
            if len(s_vals) < 20:
                continue
            r, _ = stats.pearsonr(s_vals, w_vals)
            if abs(r) > abs(max_r):
                max_r = r
                max_wc = wc_col

        print(f"  {s_col:35s} {max_r:+22.3f} {max_wc:>20s}")


def main():
    # PART A: BH correction
    print("=" * 110)
    print("  PUBLICATION VALIDATION")
    print("=" * 110)

    results = correct_comparisons()
    bh_results = print_bh_results(results)

    # Save BH-corrected results
    if bh_results:
        path = os.path.join(HERE, "results", "all_comparisons_bh_corrected.csv")
        keys = ["analysis", "comparison", "metric", "cohens_d", "p_value", "p_adj_bh", "sig_original", "sig_bh"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(bh_results)
        print(f"\n  Saved BH-corrected results to {path}")

    # PART B: Null models (needs spaCy)
    nlp = load_nlp()
    null_tea = run_null_model_tea(nlp, n_sample=25)
    null_rqa = run_null_model_rqa(nlp, n_sample=25)

    # Save null model results
    if null_tea:
        path = os.path.join(HERE, "results", "null_model_tea.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(null_tea[0].keys()))
            w.writeheader()
            w.writerows(null_tea)

    if null_rqa:
        path = os.path.join(HERE, "results", "null_model_rqa.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(null_rqa[0].keys()))
            w.writeheader()
            w.writerows(null_rqa)

    # PART C: Added value
    run_added_value_test()

    print("\n" + "=" * 110)
    print("  VALIDATION COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
