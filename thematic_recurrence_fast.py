"""
Fast thematic recurrence analysis.
Uses the already-extracted TEA metrics + spaCy doc vectors.
Does NOT re-extract TEA from scratch.

Metrics:
1. Semantic recurrence: cosine similarity between text windows (spaCy vectors)
2. Agent/Target HHI from existing TEA CSVs (re-extract only agent/target lists)
3. Node recurrence from existing GraphML files
"""
import os
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import spacy
import networkx as nx
from scipy import stats
from collections import Counter

HERE = os.path.dirname(__file__)


def load_nlp():
    print("Loading en_core_web_lg...")
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    return nlp


def semantic_recurrence_windows(text, nlp, n_windows=5):
    """Cosine similarity between consecutive text windows using spaCy vectors."""
    words = text.split()
    if len(words) < n_windows * 30:
        return None, None
    chunk_size = len(words) // n_windows
    vecs = []
    for i in range(n_windows):
        start = i * chunk_size
        end = start + chunk_size if i < n_windows - 1 else len(words)
        window_text = " ".join(words[start:end])
        doc = nlp(window_text)
        if doc.vector_norm > 0:
            vecs.append(doc.vector / doc.vector_norm)

    if len(vecs) < 2:
        return None, None

    # Mean pairwise cosine similarity (overall semantic recurrence)
    all_sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            all_sims.append(float(np.dot(vecs[i], vecs[j])))
    mean_sim = np.mean(all_sims)

    # Adjacent window similarity (local recurrence)
    adj_sims = []
    for i in range(len(vecs) - 1):
        adj_sims.append(float(np.dot(vecs[i], vecs[i + 1])))
    mean_adj = np.mean(adj_sims)

    return mean_sim, mean_adj


def node_recurrence_from_graph(graph_path, text, n_windows=5):
    """Check which graph nodes recur across text windows."""
    if not os.path.exists(graph_path):
        return None

    G = nx.read_graphml(graph_path)
    graph_nodes = set(G.nodes())
    if len(graph_nodes) < 5:
        return None

    words = text.lower().split()
    if len(words) < n_windows * 30:
        return None

    chunk_size = len(words) // n_windows
    window_hits = []
    for i in range(n_windows):
        start = i * chunk_size
        end = start + chunk_size if i < n_windows - 1 else len(words)
        window_words = set(words[start:end])
        hits = graph_nodes & window_words
        window_hits.append(hits)

    all_found = set()
    for wh in window_hits:
        all_found |= wh
    if not all_found:
        return None

    recurring = sum(1 for node in all_found
                    if sum(1 for wh in window_hits if node in wh) > 1)
    return recurring / len(all_found)


def agent_target_hhi_from_tea(tea_csv_path, slug):
    """Extract agent/target HHI from existing TEA metrics."""
    df = pd.read_csv(tea_csv_path)
    row = df[df["slug"] == slug]
    if row.empty:
        return None, None

    top_edges = str(row.iloc[0].get("top_edges", ""))
    # Can't reliably compute HHI from just top_edges
    # We need to re-extract from the graph
    return None, None


def agent_target_hhi_from_graph(graph_path):
    """Compute HHI for agent and target distributions from graph."""
    if not os.path.exists(graph_path):
        return None, None

    G = nx.read_graphml(graph_path)

    agents = {}
    targets = {}
    for n, data in G.nodes(data=True):
        role = data.get("tea_role", "")
        deg = G.degree(n, weight="weight")
        if role == "Agent":
            agents[n] = deg
        elif role == "Target":
            targets[n] = deg

    def hhi(dist):
        if not dist:
            return None
        total = sum(dist.values())
        if total == 0:
            return None
        shares = [v / total for v in dist.values()]
        return sum(s ** 2 for s in shares)

    return hhi(agents), hhi(targets)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def process_corpus(name, texts_dir, graph_dir, metadata_path, nlp, max_n=152):
    import random
    random.seed(42)

    meta = []
    with open(metadata_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta.append(row)

    # Filter control
    if "word_count" in meta[0]:
        meta = [r for r in meta if int(r.get("word_count", 0)) >= 200]

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
        if len(text.split()) < 200 or len(text) > 45000:
            continue

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{name}] [{i+1}/{len(meta)}]")

        # Semantic recurrence (fast: just spaCy vectors)
        sem_all, sem_adj = semantic_recurrence_windows(text, nlp, n_windows=5)

        # Node recurrence from existing graph
        graph_path = os.path.join(graph_dir, f"{slug}.graphml")
        nrr = node_recurrence_from_graph(graph_path, text, n_windows=5)

        # Agent/Target HHI from graph
        agent_hhi, target_hhi = agent_target_hhi_from_graph(graph_path)

        results.append({
            "slug": slug,
            "n_words": len(text.split()),
            "semantic_recurrence_all": sem_all,
            "semantic_recurrence_adj": sem_adj,
            "node_recurrence_rate": nrr,
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
        os.path.join(HERE, "results"),
        os.path.join(HERE, "metadata.csv"),
        nlp
    )

    print("\nProcessing Control corpus...")
    # Control doesn't have GraphML files, so we skip graph-based metrics for it
    # and only compute semantic recurrence
    ctrl = process_corpus(
        "Control",
        os.path.join(HERE, "control_gutenberg", "texts"),
        os.path.join(HERE, "control_gutenberg", "results"),  # no GraphML here
        os.path.join(HERE, "control_gutenberg", "metadata.csv"),
        nlp
    )

    metrics = [
        ("semantic_recurrence_all", "Semantic Recurrence (all pairs)"),
        ("semantic_recurrence_adj", "Semantic Recurrence (adjacent)"),
        ("node_recurrence_rate", "Node Recurrence Rate"),
        ("agent_hhi", "Agent Concentration (HHI)"),
        ("target_hhi", "Target Concentration (HHI)"),
    ]

    print("\n" + "=" * 100)
    print(f"{'Metric':42s} {'OCD (M+/-SD)':18s} {'Ctrl (M+/-SD)':18s} {'U':>8s} {'p':>10s} {'d':>7s} {'Eff':>10s}")
    print("=" * 100)

    for col, label in metrics:
        ocd_vals = np.array([r[col] for r in ocd if r[col] is not None])
        ctrl_vals = np.array([r[col] for r in ctrl if r[col] is not None])

        ocd_str = f"{np.mean(ocd_vals):.4f}+/-{np.std(ocd_vals):.4f}" if len(ocd_vals) > 0 else "N/A"
        ctrl_str = f"{np.mean(ctrl_vals):.4f}+/-{np.std(ctrl_vals):.4f}" if len(ctrl_vals) > 0 else "N/A"

        if len(ocd_vals) < 5 or len(ctrl_vals) < 5:
            print(f"  {label:40s} {ocd_str:18s} {ctrl_str:18s}  (OCD-only or insufficient data)")
            continue

        u, p = stats.mannwhitneyu(ocd_vals, ctrl_vals, alternative="two-sided")
        d = cohens_d(ocd_vals, ctrl_vals)
        d_abs = abs(d)
        eff = "negligible" if d_abs < 0.2 else "small" if d_abs < 0.5 else "medium" if d_abs < 0.8 else "large"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"  {label:40s} {ocd_str:18s} {ctrl_str:18s}  "
              f"{u:8.0f} {p:10.6f}{sig:3s} {d:+7.3f} {eff:>10s}")

    print("=" * 100)
    print(f"\n  OCD: n={len(ocd)}, Control: n={len(ctrl)}")

    # Save
    for name, data in [("ocd", ocd), ("control", ctrl)]:
        path = os.path.join(HERE, "results", f"thematic_recurrence_{name}.csv")
        if data:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    print("  Saved to results/thematic_recurrence_*.csv")


if __name__ == "__main__":
    main()
