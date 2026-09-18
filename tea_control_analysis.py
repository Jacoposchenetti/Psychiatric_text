"""
TEA extraction and analysis on the Gutenberg control corpus.
Same pipeline as tea_ocd_analysis.py for fair comparison.
"""
import sys
import os
import csv
import time
import warnings
import random
warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\Users\jsche\Desktop\NLP\TEA_Networks")

import spacy
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter

HERE = os.path.dirname(__file__)
CONTROL_DIR = os.path.join(HERE, "control_gutenberg")
TEXTS_DIR = os.path.join(CONTROL_DIR, "texts")
OUT_DIR = os.path.join(CONTROL_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)


def load_nlp():
    print("Loading en_core_web_lg...")
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    return nlp


def extract_tea_for_text(text, nlp):
    from teanets.svo_extraction import extract_svos
    doc = nlp(text)
    try:
        df = extract_svos(doc)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def build_tea_graph(df):
    G = nx.DiGraph()
    if df.empty:
        return G
    for _, row in df.iterrows():
        n1 = str(row["Node 1"]).strip().lower()
        tea1 = str(row["TEA"]).strip()
        n2 = str(row["Node 2"]).strip().lower()
        tea2 = str(row["TEA2"]).strip()
        if not n1 or not n2 or n1 == "nan" or n2 == "nan":
            continue
        if G.has_edge(n1, n2):
            G[n1][n2]["weight"] += 1
        else:
            G.add_edge(n1, n2, weight=1, tea_from=tea1, tea_to=tea2)
        if "tea_role" not in G.nodes[n1]:
            G.nodes[n1]["tea_role"] = tea1
        if "tea_role" not in G.nodes[n2]:
            G.nodes[n2]["tea_role"] = tea2
    return G


def compute_metrics(G, df, text):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        return None

    words = text.split()
    n_words = len(words)
    n_triplets = len(df) if not df.empty else 0
    density = nx.density(G)

    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    total_deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes}
    avg_degree = np.mean(list(total_deg.values())) if total_deg else 0

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    repetition_idx = sum(1 for w in weights if w > 1) / max(len(weights), 1)
    max_weight = max(weights) if weights else 0
    mean_weight = np.mean(weights) if weights else 0

    if n_nodes > 1:
        sccs = list(nx.strongly_connected_components(G))
        lscc = max(len(c) for c in sccs) if sccs else 0
        lscc_frac = lscc / n_nodes
        wccs = list(nx.weakly_connected_components(G))
        lwcc = max(len(c) for c in wccs) if wccs else 0
        lwcc_frac = lwcc / n_nodes
    else:
        lscc_frac = lwcc_frac = 1.0

    Gu = G.to_undirected()
    avg_clustering = nx.average_clustering(Gu) if n_nodes > 2 else 0

    agents = [n for n, d in G.nodes(data=True) if d.get("tea_role") == "Agent"]
    events = [n for n, d in G.nodes(data=True) if d.get("tea_role") == "Event"]
    targets = [n for n, d in G.nodes(data=True) if d.get("tea_role") == "Target"]

    first_person = {"i", "me", "my", "myself", "we", "us", "our"}
    fp_agent_count = sum(1 for _, row in df.iterrows()
                         if str(row.get("Node 1", "")).strip().lower() in first_person
                         and str(row.get("TEA", "")).strip() == "Agent") if not df.empty else 0
    fp_ratio = fp_agent_count / max(n_triplets, 1)

    node_forms = [str(row["Node 1"]).lower() for _, row in df.iterrows()] + \
                 [str(row["Node 2"]).lower() for _, row in df.iterrows()] if not df.empty else []
    ttr = len(set(node_forms)) / max(len(node_forms), 1)

    top_edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)[:10]
    top_edges_str = "; ".join(f"{u}->{v}({d['weight']})" for u, v, d in top_edges)

    return {
        "n_words": n_words,
        "n_triplets": n_triplets,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": round(density, 4),
        "avg_degree": round(avg_degree, 2),
        "repetition_idx": round(repetition_idx, 4),
        "max_edge_weight": max_weight,
        "mean_edge_weight": round(mean_weight, 2),
        "lscc_frac": round(lscc_frac, 4),
        "lwcc_frac": round(lwcc_frac, 4),
        "avg_clustering": round(avg_clustering, 4),
        "n_agents": len(agents),
        "n_events": len(events),
        "n_targets": len(targets),
        "first_person_ratio": round(fp_ratio, 4),
        "node_ttr": round(ttr, 4),
        "top_edges": top_edges_str,
    }


def main():
    nlp = load_nlp()

    meta = []
    with open(os.path.join(CONTROL_DIR, "metadata.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["word_count"] = int(row["word_count"])
            meta.append(row)

    # Filter to 300-5000 words and sample 152 to match OCD corpus size
    filtered = [r for r in meta if 300 <= r["word_count"] <= 5000]
    sample = random.sample(filtered, min(152, len(filtered)))
    print(f"Processing {len(sample)} control segments\n")

    results = []
    t0 = time.time()
    for i, row in enumerate(sample):
        slug = row["slug"]
        txt_path = os.path.join(TEXTS_DIR, f"{slug}.txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        print(f"[{i+1}/{len(sample)}] {slug} ({len(text.split())} words)")
        df = extract_tea_for_text(text, nlp)
        G = build_tea_graph(df)
        metrics = compute_metrics(G, df, text)
        if metrics is None:
            continue

        metrics["slug"] = slug
        metrics["source"] = row.get("source", "")
        results.append(metrics)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(results)} segments in {elapsed:.0f}s")

    out_csv = os.path.join(OUT_DIR, "tea_metrics.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"Metrics saved to {out_csv}")

    if results:
        df_m = pd.DataFrame(results)
        print("\n=== CONTROL CORPUS SUMMARY ===")
        for col in ["n_triplets", "n_nodes", "n_edges", "density", "repetition_idx",
                     "mean_edge_weight", "lscc_frac", "avg_clustering",
                     "first_person_ratio", "node_ttr"]:
            vals = df_m[col].dropna()
            print(f"  {col:25s}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
                  f"median={vals.median():.3f}")


if __name__ == "__main__":
    main()
