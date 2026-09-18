"""
Deep structural analysis of TEA networks from OCD narratives.
Betweenness centrality, community detection, hub/authority scores,
degree distributions, loop detection, and topic recurrence patterns.
"""
import os
import csv
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
STRUCT_DIR = os.path.join(RESULTS_DIR, "structure")
os.makedirs(STRUCT_DIR, exist_ok=True)


def load_graph(slug):
    path = os.path.join(RESULTS_DIR, f"{slug}.graphml")
    if not os.path.exists(path):
        return None
    return nx.read_graphml(path)


def analyze_centrality(G):
    """Betweenness, closeness, eigenvector centrality."""
    if G.number_of_nodes() < 3:
        return {}

    bc = nx.betweenness_centrality(G, weight="weight")
    top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]

    # PageRank as proxy for eigenvector centrality on directed graphs
    try:
        pr = nx.pagerank(G, weight="weight")
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        top_pr = []

    # HITS hub/authority
    try:
        hubs, auths = nx.hits(G, max_iter=300)
        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:10]
        top_auths = sorted(auths.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        top_hubs, top_auths = [], []

    bc_values = list(bc.values())
    return {
        "mean_betweenness": np.mean(bc_values),
        "max_betweenness": max(bc_values),
        "gini_betweenness": gini(bc_values),
        "top_betweenness": [(n, round(v, 4)) for n, v in top_bc[:5]],
        "top_pagerank": [(n, round(v, 4)) for n, v in top_pr[:5]],
        "top_hubs": [(n, round(v, 4)) for n, v in top_hubs[:5]],
        "top_authorities": [(n, round(v, 4)) for n, v in top_auths[:5]],
    }


def gini(values):
    """Compute Gini coefficient for a list of values."""
    arr = np.array(sorted(values), dtype=float)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * arr) / (n * arr.sum())) - (n + 1) / n


def analyze_communities(G):
    """Community detection via Louvain on undirected projection."""
    Gu = G.to_undirected()
    if Gu.number_of_nodes() < 3:
        return {}

    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(Gu, weight="weight", seed=42)
    except Exception:
        communities = list(nx.connected_components(Gu))

    n_communities = len(communities)
    sizes = sorted([len(c) for c in communities], reverse=True)

    # Modularity
    try:
        modularity = nx.algorithms.community.modularity(Gu, communities, weight="weight")
    except Exception:
        modularity = None

    # Get top nodes per community
    community_tops = []
    for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:5]):
        subg = G.subgraph(comm)
        deg = dict(subg.degree(weight="weight"))
        top = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:5]
        community_tops.append({
            "id": i,
            "size": len(comm),
            "top_nodes": [(n, round(d, 1)) for n, d in top],
        })

    return {
        "n_communities": n_communities,
        "community_sizes": sizes[:10],
        "modularity": round(modularity, 4) if modularity is not None else None,
        "top_communities": community_tops,
    }


def analyze_loops(G):
    """Detect cycles and repetitive patterns in the graph."""
    if G.number_of_nodes() < 3:
        return {}

    # Simple cycles (capped for performance)
    try:
        cycles = []
        for cycle in nx.simple_cycles(G, length_bound=5):
            if len(cycle) >= 2:
                cycles.append(cycle)
            if len(cycles) > 500:
                break
    except Exception:
        cycles = []

    cycle_lengths = [len(c) for c in cycles]

    # Self-loops (node -> ... -> node)
    self_loops = [(u, v, d["weight"]) for u, v, d in G.edges(data=True) if u == v]

    # Reciprocal edges (A->B and B->A)
    reciprocal = []
    for u, v in G.edges():
        if G.has_edge(v, u) and u < v:
            w1 = G[u][v].get("weight", 1)
            w2 = G[v][u].get("weight", 1)
            reciprocal.append((u, v, w1 + w2))
    reciprocal.sort(key=lambda x: x[2], reverse=True)

    # Strongly connected components (obsessive loops)
    sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
    scc_sizes = sorted([len(c) for c in sccs], reverse=True)

    return {
        "n_cycles_up_to_5": len(cycles),
        "cycle_length_dist": Counter(cycle_lengths),
        "n_self_loops": len(self_loops),
        "n_reciprocal_edges": len(reciprocal),
        "top_reciprocal": reciprocal[:10],
        "n_sccs": len(sccs),
        "scc_sizes": scc_sizes[:10],
    }


def analyze_degree_distribution(G):
    """Degree distribution and power-law fit indicators."""
    in_deg = [d for _, d in G.in_degree(weight="weight")]
    out_deg = [d for _, d in G.out_degree(weight="weight")]
    total_deg = [i + o for i, o in zip(in_deg, out_deg)]

    if not total_deg:
        return {}

    return {
        "max_in_degree": max(in_deg),
        "max_out_degree": max(out_deg),
        "mean_total_degree": round(np.mean(total_deg), 2),
        "std_total_degree": round(np.std(total_deg), 2),
        "skewness_degree": round(float(pd.Series(total_deg).skew()), 3),
        "gini_degree": round(gini(total_deg), 4),
    }


def analyze_tea_roles(G):
    """Analyze patterns by TEA role (Agent, Event, Target)."""
    agents = {n: d for n, d in G.nodes(data=True) if d.get("tea_role") == "Agent"}
    events = {n: d for n, d in G.nodes(data=True) if d.get("tea_role") == "Event"}
    targets = {n: d for n, d in G.nodes(data=True) if d.get("tea_role") == "Target"}

    # Agent diversity
    agent_deg = {n: G.out_degree(n, weight="weight") for n in agents}
    event_deg = {n: G.degree(n, weight="weight") for n in events}
    target_deg = {n: G.in_degree(n, weight="weight") for n in targets}

    # Top agents, events, targets
    top_agents = sorted(agent_deg.items(), key=lambda x: x[1], reverse=True)[:10]
    top_events = sorted(event_deg.items(), key=lambda x: x[1], reverse=True)[:10]
    top_targets = sorted(target_deg.items(), key=lambda x: x[1], reverse=True)[:10]

    # Agent concentration: how much does the top agent dominate?
    if agent_deg:
        agent_vals = list(agent_deg.values())
        top_agent_share = max(agent_vals) / max(sum(agent_vals), 1)
    else:
        top_agent_share = 0

    return {
        "n_agents": len(agents),
        "n_events": len(events),
        "n_targets": len(targets),
        "top_agents": top_agents[:5],
        "top_events": top_events[:5],
        "top_targets": top_targets[:5],
        "agent_concentration": round(top_agent_share, 4),
        "gini_agent_degree": round(gini(list(agent_deg.values())), 4) if agent_deg else 0,
    }


def main():
    meta = pd.read_csv(os.path.join(HERE, "results", "tea_metrics.csv"))
    slugs = meta["slug"].tolist()

    all_struct = []
    corpus_centrality = defaultdict(list)
    corpus_communities = []
    corpus_loops = defaultdict(list)

    for i, slug in enumerate(slugs):
        G = load_graph(slug)
        if G is None or G.number_of_nodes() < 5:
            continue

        print(f"[{i+1}/{len(slugs)}] {slug} ({G.number_of_nodes()} nodes)")

        centr = analyze_centrality(G)
        comms = analyze_communities(G)
        loops = analyze_loops(G)
        deg = analyze_degree_distribution(G)
        roles = analyze_tea_roles(G)

        row = {"slug": slug}
        for d in [centr, comms, loops, deg, roles]:
            for k, v in d.items():
                if isinstance(v, (int, float, np.floating)):
                    row[k] = v
                    if k in ("mean_betweenness", "max_betweenness", "gini_betweenness",
                             "n_communities", "modularity", "n_cycles_up_to_5",
                             "n_reciprocal_edges", "n_sccs", "gini_degree",
                             "skewness_degree", "agent_concentration", "gini_agent_degree"):
                        corpus_centrality[k].append(v)

        # Aggregate top nodes across corpus
        for n, v in centr.get("top_betweenness", []):
            corpus_centrality["bc_nodes"].append(n)
        for n, v in centr.get("top_hubs", []):
            corpus_centrality["hub_nodes"].append(n)
        for n, v in centr.get("top_authorities", []):
            corpus_centrality["auth_nodes"].append(n)

        corpus_loops["n_cycles"].append(loops.get("n_cycles_up_to_5", 0))
        corpus_loops["n_reciprocal"].append(loops.get("n_reciprocal_edges", 0))

        all_struct.append(row)

    # Save per-story structural metrics
    if all_struct:
        scalar_keys = [k for k in all_struct[0].keys()
                       if isinstance(all_struct[0][k], (int, float, np.floating, type(None)))]
        out_csv = os.path.join(STRUCT_DIR, "structural_metrics.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
            w.writeheader()
            for row in all_struct:
                clean = {k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in row.items() if k in scalar_keys}
                w.writerow(clean)
        print(f"\nStructural metrics saved to {out_csv}")

    # Corpus-level summary
    print("\n" + "=" * 60)
    print("CORPUS-LEVEL STRUCTURAL ANALYSIS")
    print("=" * 60)

    for metric in ["mean_betweenness", "max_betweenness", "gini_betweenness",
                   "n_communities", "modularity", "n_cycles_up_to_5",
                   "n_reciprocal_edges", "n_sccs", "gini_degree",
                   "skewness_degree", "agent_concentration", "gini_agent_degree"]:
        vals = [v for v in corpus_centrality.get(metric, []) if v is not None]
        if vals:
            print(f"  {metric:30s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"median={np.median(vals):.4f}")

    # Most common high-centrality nodes
    print("\n--- Most common high-betweenness nodes across corpus ---")
    bc_counter = Counter(corpus_centrality.get("bc_nodes", []))
    for node, cnt in bc_counter.most_common(20):
        print(f"  {node:30s}  {cnt}")

    print("\n--- Most common hub nodes (HITS) ---")
    hub_counter = Counter(corpus_centrality.get("hub_nodes", []))
    for node, cnt in hub_counter.most_common(20):
        print(f"  {node:30s}  {cnt}")

    print("\n--- Most common authority nodes (HITS) ---")
    auth_counter = Counter(corpus_centrality.get("auth_nodes", []))
    for node, cnt in auth_counter.most_common(20):
        print(f"  {node:30s}  {cnt}")

    print("\nDone.")


if __name__ == "__main__":
    main()
