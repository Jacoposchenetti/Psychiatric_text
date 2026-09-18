"""
Measure actual thematic recurrence in OCD vs Control narratives.

Three concrete metrics:
1. Node recurrence rate: fraction of TEA nodes that appear in multiple
   non-overlapping windows of the text (same concept returning later).
2. Semantic recurrence (cosine similarity between text windows):
   how similar are different parts of the same narrative?
3. Agent/Target concentration: how few distinct agents/targets account
   for most of the triplets (Herfindahl index).
"""
import sys
import os
import csv
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\Users\jsche\Desktop\NLP\TEA_Networks")

import numpy as np
import pandas as pd
import spacy
from scipy import stats
from collections import Counter

HERE = os.path.dirname(__file__)


def load_nlp():
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    return nlp


def split_into_windows(text, n_windows=5):
    words = text.split()
    if len(words) < n_windows * 20:
        return None
    chunk_size = len(words) // n_windows
    windows = []
    for i in range(n_windows):
        start = i * chunk_size
        end = start + chunk_size if i < n_windows - 1 else len(words)
        windows.append(" ".join(words[start:end]))
    return windows


def extract_tea_nodes_per_window(windows, nlp):
    """Extract TEA nodes from each window separately."""
    from teanets.svo_extraction import extract_svos
    window_nodes = []
    for w in windows:
        doc = nlp(w)
        try:
            df = extract_svos(doc)
            if df is not None and len(df) > 0:
                nodes = set()
                for _, row in df.iterrows():
                    n1 = str(row["Node 1"]).strip().lower()
                    n2 = str(row["Node 2"]).strip().lower()
                    if n1 and n1 != "nan":
                        nodes.add(n1)
                    if n2 and n2 != "nan":
                        nodes.add(n2)
                window_nodes.append(nodes)
            else:
                window_nodes.append(set())
        except Exception:
            window_nodes.append(set())
    return window_nodes


def node_recurrence_rate(window_nodes):
    """Fraction of unique nodes that appear in more than one window."""
    all_nodes = set()
    for wn in window_nodes:
        all_nodes |= wn
    if not all_nodes:
        return 0.0
    recurring = 0
    for node in all_nodes:
        count = sum(1 for wn in window_nodes if node in wn)
        if count > 1:
            recurring += 1
    return recurring / len(all_nodes)


def semantic_recurrence(windows, nlp):
    """Mean pairwise cosine similarity between window embeddings."""
    vecs = []
    for w in windows:
        doc = nlp(w)
        if doc.vector_norm > 0:
            vecs.append(doc.vector / doc.vector_norm)
    if len(vecs) < 2:
        return None
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(float(np.dot(vecs[i], vecs[j])))
    return np.mean(sims)


def agent_target_concentration(text, nlp):
    """Herfindahl-Hirschman Index for agent and target distributions.
    Higher HHI = more concentrated (fewer entities dominate)."""
    from teanets.svo_extraction import extract_svos
    doc = nlp(text)
    try:
        df = extract_svos(doc)
    except Exception:
        return None, None
    if df is None or len(df) == 0:
        return None, None

    agents = [str(row["Node 1"]).strip().lower()
              for _, row in df.iterrows()
              if str(row.get("TEA", "")).strip() == "Agent"]
    targets = [str(row["Node 2"]).strip().lower()
               for _, row in df.iterrows()
               if str(row.get("TEA2", "")).strip() == "Target"]

    def hhi(items):
        if not items:
            return None
        counts = Counter(items)
        total = sum(counts.values())
        shares = [c / total for c in counts.values()]
        return sum(s ** 2 for s in shares)

    return hhi(agents), hhi(targets)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def process_corpus(name, texts_dir, metadata_path, nlp, max_n=152):
    """Process a corpus and return recurrence metrics."""
    import random
    random.seed(42)

    meta = []
    with open(metadata_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta.append(row)

    if len(meta) > max_n:
        meta = random.sample(meta, max_n)

    results = []
    for i, row in enumerate(meta):
        slug = row["slug"]
        txt_path = os.path.join(texts_dir, f"{slug}.txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()
        if len(text.split()) < 200:
            continue

        print(f"  [{name}] [{i+1}/{len(meta)}] {slug}")

        windows = split_into_windows(text, n_windows=5)
        if windows is None:
            continue

        window_nodes = extract_tea_nodes_per_window(windows, nlp)
        nrr = node_recurrence_rate(window_nodes)
        sem_rec = semantic_recurrence(windows, nlp)
        agent_hhi, target_hhi = agent_target_concentration(text, nlp)

        results.append({
            "slug": slug,
            "n_words": len(text.split()),
            "node_recurrence_rate": nrr,
            "semantic_recurrence": sem_rec,
            "agent_hhi": agent_hhi,
            "target_hhi": target_hhi,
        })

    return results


def main():
    nlp = load_nlp()

    print("Processing OCD corpus...")
    ocd = process_corpus(
        "OCD",
        os.path.join(HERE, "texts"),
        os.path.join(HERE, "metadata.csv"),
        nlp
    )

    print("\nProcessing Control corpus...")
    ctrl_texts = os.path.join(HERE, "control_gutenberg", "texts")
    ctrl_meta = os.path.join(HERE, "control_gutenberg", "metadata.csv")

    # Filter control to 300-5000 words
    ctrl_meta_filtered = os.path.join(HERE, "control_gutenberg", "metadata_filtered.csv")
    rows_filt = []
    with open(ctrl_meta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if 300 <= int(row["word_count"]) <= 5000:
                rows_filt.append(row)
    with open(ctrl_meta_filtered, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_filt[0].keys()))
        w.writeheader()
        w.writerows(rows_filt)

    ctrl = process_corpus(
        "Control",
        ctrl_texts,
        ctrl_meta_filtered,
        nlp
    )

    # Compare
    metrics = [
        ("node_recurrence_rate", "Node Recurrence Rate"),
        ("semantic_recurrence", "Semantic Recurrence (cosine)"),
        ("agent_hhi", "Agent Concentration (HHI)"),
        ("target_hhi", "Target Concentration (HHI)"),
    ]

    print("\n" + "=" * 95)
    print(f"{'Metric':40s} {'OCD (M+/-SD)':18s} {'Ctrl (M+/-SD)':18s} {'U':>8s} {'p':>10s} {'d':>7s} {'Eff':>10s}")
    print("=" * 95)

    for col, label in metrics:
        ocd_vals = np.array([r[col] for r in ocd if r[col] is not None])
        ctrl_vals = np.array([r[col] for r in ctrl if r[col] is not None])

        if len(ocd_vals) < 5 or len(ctrl_vals) < 5:
            print(f"  {label:38s} insufficient data")
            continue

        ocd_m, ocd_s = np.mean(ocd_vals), np.std(ocd_vals)
        ctrl_m, ctrl_s = np.mean(ctrl_vals), np.std(ctrl_vals)
        u, p = stats.mannwhitneyu(ocd_vals, ctrl_vals, alternative="two-sided")
        d = cohens_d(ocd_vals, ctrl_vals)

        d_abs = abs(d)
        eff = "negligible" if d_abs < 0.2 else "small" if d_abs < 0.5 else "medium" if d_abs < 0.8 else "large"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"  {label:38s} {ocd_m:.4f}+/-{ocd_s:.4f}  {ctrl_m:.4f}+/-{ctrl_s:.4f}  "
              f"{u:8.0f} {p:10.6f}{sig:3s} {d:+7.3f} {eff:>10s}")

    print("=" * 95)
    print(f"\n  OCD: n={len(ocd)}, Control: n={len(ctrl)}")

    # Save
    for name, data in [("ocd", ocd), ("control", ctrl)]:
        path = os.path.join(HERE, "results", f"thematic_recurrence_{name}.csv")
        if data:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    print("  Results saved to results/thematic_recurrence_*.csv")


if __name__ == "__main__":
    main()
