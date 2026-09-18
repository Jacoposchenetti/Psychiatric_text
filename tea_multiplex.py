"""
TEA Multiplex: Multi-emotional multiplex network analysis of narratives.

Extends TEA Networks by treating each Plutchik emotion as a separate layer
in a multiplex network. Each SVO triplet is assigned to emotion layers based
on the emotional profile of the Event (verb) and context words.

Layers:
  - syntactic: all SVO edges (base TEA graph)
  - fear, anger, sadness, joy, trust, disgust, surprise, anticipation:
    edges where the Event carries that emotion (NRC EmoLex)

Multiplex metrics computed per text:
  - layer_dominance: fraction of edges in each emotional layer
  - participation_coefficient: how evenly each node spreads across layers
  - inter_layer_correlation: degree correlation between layer pairs
  - emotional_lscc: LSCC fraction per emotional layer
"""
import sys
import os
import csv
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import networkx as nx
import spacy
from scipy import stats
from nrclex import NRCLex

sys.path.insert(0, r"C:\Users\jsche\Desktop\NLP\TEA_Networks")

HERE = os.path.dirname(__file__)
random.seed(42)

PLUTCHIK = ["fear", "anger", "sadness", "joy", "trust", "disgust", "surprise", "anticipation"]

# Pre-load NRC lexicon once
_nrc = NRCLex("init")
NRC_LEX = _nrc.__lexicon__


def load_nlp():
    print("Loading en_core_web_lg...")
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    return nlp


def get_word_emotions(word):
    """Return set of Plutchik emotions for a word from NRC EmoLex."""
    emos = NRC_LEX.get(word.lower(), [])
    return set(e for e in emos if e in PLUTCHIK)


def get_triplet_emotions(event_text, node1_text, node2_text):
    """
    Determine which emotional layers a triplet belongs to.
    Primary signal: the Event (verb). Secondary: Agent and Target context.
    """
    emotions = set()
    for word in event_text.lower().split():
        emotions |= get_word_emotions(word)
    for word in node1_text.lower().split():
        emotions |= get_word_emotions(word)
    for word in node2_text.lower().split():
        emotions |= get_word_emotions(word)
    return emotions


def extract_tea_triplets(text, nlp):
    """Extract SVO triplets using TEA Networks."""
    from teanets.svo_extraction import extract_svos
    if len(text) > 49000:
        text = text[:49000]
    doc = nlp(text)
    try:
        df = extract_svos(doc)
        if df is None or len(df) == 0:
            return []
        triplets = []
        for _, row in df.iterrows():
            n1 = str(row["Node 1"]).strip().lower()
            tea1 = str(row["TEA"]).strip()
            n2 = str(row["Node 2"]).strip().lower()
            tea2 = str(row["TEA2"]).strip()
            if not n1 or not n2 or n1 == "nan" or n2 == "nan":
                continue
            event_col = row.get("Hypergraph", "")
            event_text = str(event_col) if event_col != "N/A" else ""
            triplets.append({
                "node1": n1, "tea1": tea1,
                "node2": n2, "tea2": tea2,
                "event_text": event_text,
            })
        return triplets
    except Exception:
        return []


def build_multiplex(triplets):
    """
    Build a multiplex network from TEA triplets.
    Returns dict: {"syntactic": DiGraph, "fear": DiGraph, ...}
    """
    layers = {"syntactic": nx.DiGraph()}
    for emo in PLUTCHIK:
        layers[emo] = nx.DiGraph()

    for t in triplets:
        n1, n2 = t["node1"], t["node2"]

        # Syntactic layer: all edges
        if layers["syntactic"].has_edge(n1, n2):
            layers["syntactic"][n1][n2]["weight"] += 1
        else:
            layers["syntactic"].add_edge(n1, n2, weight=1)

        # Emotional layers
        emotions = get_triplet_emotions(t["event_text"], n1, n2)
        for emo in emotions:
            G = layers[emo]
            if G.has_edge(n1, n2):
                G[n1][n2]["weight"] += 1
            else:
                G.add_edge(n1, n2, weight=1)

    return layers


def compute_layer_dominance(layers):
    """Fraction of edges in each emotional layer relative to total emotional edges."""
    total_emo_edges = sum(layers[e].number_of_edges() for e in PLUTCHIK)
    if total_emo_edges == 0:
        return {e: 0.0 for e in PLUTCHIK}
    return {e: layers[e].number_of_edges() / total_emo_edges for e in PLUTCHIK}


def compute_participation_coefficient(layers):
    """
    Multiplex participation coefficient per node.
    P_i = (M/(M-1)) * (1 - sum_alpha (k_i^alpha / o_i)^2)
    where k_i^alpha = degree in layer alpha, o_i = total degree across layers.
    P=0: node active in one layer only. P→1: equally distributed.
    """
    all_nodes = set()
    for emo in PLUTCHIK:
        all_nodes |= set(layers[emo].nodes())

    if not all_nodes:
        return 0.0

    M = len(PLUTCHIK)
    coefficients = []

    for node in all_nodes:
        degrees = []
        for emo in PLUTCHIK:
            G = layers[emo]
            if G.has_node(node):
                degrees.append(G.degree(node, weight="weight"))
            else:
                degrees.append(0)

        o_i = sum(degrees)
        if o_i == 0:
            continue

        sum_sq = sum((k / o_i) ** 2 for k in degrees)
        P_i = (M / (M - 1)) * (1 - sum_sq)
        coefficients.append(P_i)

    return np.mean(coefficients) if coefficients else 0.0


def compute_inter_layer_correlation(layers, layer_a, layer_b):
    """Pearson correlation of node degrees between two layers."""
    nodes_a = set(layers[layer_a].nodes())
    nodes_b = set(layers[layer_b].nodes())
    common = nodes_a & nodes_b

    if len(common) < 5:
        return np.nan

    deg_a = [layers[layer_a].degree(n, weight="weight") for n in common]
    deg_b = [layers[layer_b].degree(n, weight="weight") for n in common]

    if np.std(deg_a) == 0 or np.std(deg_b) == 0:
        return np.nan

    r, _ = stats.pearsonr(deg_a, deg_b)
    return r


def compute_emotional_lscc(layers):
    """LSCC fraction for each emotional layer."""
    result = {}
    for emo in PLUTCHIK:
        G = layers[emo]
        n = G.number_of_nodes()
        if n < 2:
            result[emo] = 0.0
            continue
        sccs = list(nx.strongly_connected_components(G))
        lscc = max(len(c) for c in sccs) if sccs else 0
        result[emo] = lscc / n
    return result


def compute_first_person_emotional_profile(triplets):
    """
    For first-person agent triplets (I/me/my/myself/we/us/our),
    compute which emotions dominate.
    """
    fp = {"i", "me", "my", "myself", "we", "us", "our"}
    emo_counts = {e: 0 for e in PLUTCHIK}
    fp_total = 0

    for t in triplets:
        if t["node1"] in fp and t["tea1"] == "Agent":
            fp_total += 1
            emotions = get_triplet_emotions(t["event_text"], t["node1"], t["node2"])
            for emo in emotions:
                emo_counts[emo] += 1

    if fp_total == 0:
        return {f"fp_{e}": 0.0 for e in PLUTCHIK}
    return {f"fp_{e}": emo_counts[e] / fp_total for e in PLUTCHIK}


def analyze_text(text, nlp):
    """Full multiplex analysis of a single text."""
    triplets = extract_tea_triplets(text, nlp)
    if len(triplets) < 5:
        return None

    layers = build_multiplex(triplets)

    # Basic counts
    n_syntactic_edges = layers["syntactic"].number_of_edges()
    n_syntactic_nodes = layers["syntactic"].number_of_nodes()
    if n_syntactic_nodes == 0:
        return None

    # Layer dominance
    dominance = compute_layer_dominance(layers)

    # Participation coefficient
    participation = compute_participation_coefficient(layers)

    # Inter-layer correlations (syntactic vs each emotion)
    correlations = {}
    for emo in PLUTCHIK:
        r = compute_inter_layer_correlation(layers, "syntactic", emo)
        correlations[f"corr_synt_{emo}"] = r

    # Emotional LSCC
    emo_lscc = compute_emotional_lscc(layers)

    # First-person emotional profile
    fp_profile = compute_first_person_emotional_profile(triplets)

    # Emotional layer coverage: fraction of syntactic nodes present in each emo layer
    coverage = {}
    for emo in PLUTCHIK:
        emo_nodes = set(layers[emo].nodes())
        synt_nodes = set(layers["syntactic"].nodes())
        coverage[f"coverage_{emo}"] = len(emo_nodes & synt_nodes) / max(len(synt_nodes), 1)

    # Dominant emotion
    dom_emo = max(PLUTCHIK, key=lambda e: dominance[e])
    dom_ratio = dominance[dom_emo]

    # Emotional concentration (Herfindahl index on layer dominance)
    hhi = sum(v ** 2 for v in dominance.values())

    result = {
        "n_words": len(text.split()),
        "n_triplets": len(triplets),
        "n_nodes": n_syntactic_nodes,
        "n_edges": n_syntactic_edges,
        "participation_coeff": round(participation, 4),
        "dominant_emotion": dom_emo,
        "dominant_ratio": round(dom_ratio, 4),
        "emotional_concentration": round(hhi, 4),
    }

    for emo in PLUTCHIK:
        result[f"dom_{emo}"] = round(dominance[emo], 4)
        result[f"lscc_{emo}"] = round(emo_lscc[emo], 4)
        result[f"coverage_{emo}"] = round(coverage[f"coverage_{emo}"], 4)

    result.update({k: round(v, 4) if not np.isnan(v) else None for k, v in correlations.items()})
    result.update({k: round(v, 4) for k, v in fp_profile.items()})

    return result


def process_corpus(name, texts_dir, metadata_path, nlp, max_n=200):
    """Process a corpus and return multiplex metrics."""
    meta = []
    with open(metadata_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta.append(row)

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
        if len(text.split()) < 200:
            continue

        metrics = analyze_text(text, nlp)
        if metrics is None:
            continue

        metrics["slug"] = slug
        results.append(metrics)

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{name}] [{i+1}/{len(meta)}] dom={metrics['dominant_emotion']}, "
                  f"P={metrics['participation_coeff']:.3f}, "
                  f"fear={metrics['dom_fear']:.3f}, sad={metrics['dom_sadness']:.3f}")

    return results


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def effect_label(d):
    d = abs(d)
    if d < 0.2: return "negl."
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "LARGE"


def match_by_word_count(group_a, group_b, tolerance=0.35):
    b_words = np.array([r["n_words"] for r in group_b])
    matched_a, matched_b = [], []
    indices_a = list(range(len(group_a)))
    random.shuffle(indices_a)
    used_b = set()

    for ia in indices_a:
        wc_a = group_a[ia]["n_words"]
        best_ib, best_diff = None, float("inf")
        for ib in range(len(group_b)):
            if ib in used_b:
                continue
            diff = abs(b_words[ib] - wc_a)
            if diff < best_diff:
                best_diff = diff
                best_ib = ib
        if best_ib is not None and best_diff / max(wc_a, 1) <= tolerance:
            used_b.add(best_ib)
            matched_a.append(group_a[ia])
            matched_b.append(group_b[best_ib])

    return matched_a, matched_b


def compare_groups(name_a, data_a, name_b, data_b, metrics):
    """Length-matched comparison between two groups."""
    matched_a, matched_b = match_by_word_count(data_a, data_b)

    if len(matched_a) < 15:
        print(f"  Too few matched pairs ({len(matched_a)}) for {name_a} vs {name_b}")
        return []

    wc_a = np.array([r["n_words"] for r in matched_a])
    wc_b = np.array([r["n_words"] for r in matched_b])
    d_wc = cohens_d(wc_a, wc_b)

    print(f"\n  {name_a} vs {name_b}: {len(matched_a)} matched pairs")
    print(f"  Word count match: d={d_wc:+.3f}")
    print(f"\n  {'Metric':40s} {name_a:>10s} {name_b:>10s} {'d':>8s} {'p':>10s} {'Eff':>8s}")
    print("  " + "-" * 90)

    results = []
    for col, label in metrics:
        vals_a = np.array([r[col] for r in matched_a if r.get(col) is not None])
        vals_b = np.array([r[col] for r in matched_b if r.get(col) is not None])

        if len(vals_a) < 10 or len(vals_b) < 10:
            continue

        u, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        d = cohens_d(vals_a, vals_b)
        eff = effect_label(d)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"  {label:40s} {np.mean(vals_a):10.4f} {np.mean(vals_b):10.4f} "
              f"{d:+8.3f} {p:10.6f}{sig:3s} {eff:>8s}")

        results.append({
            "comparison": f"{name_a} vs {name_b}",
            "metric": label,
            "a_mean": round(float(np.mean(vals_a)), 4),
            "b_mean": round(float(np.mean(vals_b)), 4),
            "cohens_d": round(d, 3),
            "p_value": round(p, 6),
            "effect": eff,
            "n_pairs": len(matched_a),
            "significant": p < 0.05,
        })

    return results


def main():
    nlp = load_nlp()

    print("=" * 100)
    print("  TEA MULTIPLEX: Multi-emotional network analysis")
    print("  Plutchik emotions as separate network layers")
    print("=" * 100)

    # Process each corpus
    print("\nProcessing OCD Stories...")
    t0 = time.time()
    ocd = process_corpus("OCD",
                         os.path.join(HERE, "texts"),
                         os.path.join(HERE, "metadata.csv"),
                         nlp, max_n=153)
    print(f"  OCD: {len(ocd)} texts in {time.time()-t0:.0f}s")

    print("\nProcessing Cancer Patient Stories...")
    t0 = time.time()
    cancer = process_corpus("Cancer",
                            os.path.join(HERE, "control_patient_stories", "texts"),
                            os.path.join(HERE, "control_patient_stories", "metadata.csv"),
                            nlp, max_n=152)
    print(f"  Cancer: {len(cancer)} texts in {time.time()-t0:.0f}s")

    print("\nProcessing Depression/PTSD Reddit...")
    t0 = time.time()
    mh = process_corpus("Dep/PTSD",
                        os.path.join(HERE, "control_mental_health", "texts"),
                        os.path.join(HERE, "control_mental_health", "metadata.csv"),
                        nlp, max_n=200)
    print(f"  Dep/PTSD: {len(mh)} texts in {time.time()-t0:.0f}s")

    # Save per-group metrics
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for name, data in [("ocd", ocd), ("cancer", cancer), ("mh", mh)]:
        if data:
            path = os.path.join(HERE, "results", f"multiplex_{name}.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
            print(f"  Saved {path}")

    # Print group summaries
    print("\n" + "=" * 100)
    print("  GROUP EMOTIONAL PROFILES")
    print("=" * 100)

    for name, data in [("OCD", ocd), ("Cancer", cancer), ("Dep/PTSD", mh)]:
        print(f"\n  {name} (n={len(data)}):")
        print(f"  {'Emotion':15s} {'Dominance':>10s} {'FP-ratio':>10s} {'LSCC':>10s} {'Coverage':>10s}")
        print("  " + "-" * 60)
        for emo in PLUTCHIK:
            dom = np.mean([r[f"dom_{emo}"] for r in data])
            fp = np.mean([r[f"fp_{emo}"] for r in data])
            lscc = np.mean([r[f"lscc_{emo}"] for r in data])
            cov = np.mean([r[f"coverage_{emo}"] for r in data])
            print(f"  {emo:15s} {dom:10.4f} {fp:10.4f} {lscc:10.4f} {cov:10.4f}")
        p_coeff = np.mean([r["participation_coeff"] for r in data])
        hhi = np.mean([r["emotional_concentration"] for r in data])
        print(f"  {'Participation':15s} {p_coeff:10.4f}")
        print(f"  {'Concentration':15s} {hhi:10.4f}")

    # Comparison metrics
    metrics = [
        ("participation_coeff", "Participation Coefficient"),
        ("emotional_concentration", "Emotional Concentration (HHI)"),
        ("dominant_ratio", "Dominant Emotion Ratio"),
    ]

    for emo in PLUTCHIK:
        metrics.append((f"dom_{emo}", f"Layer Dominance: {emo}"))

    for emo in PLUTCHIK:
        metrics.append((f"fp_{emo}", f"First-Person {emo}"))

    for emo in PLUTCHIK:
        metrics.append((f"lscc_{emo}", f"LSCC Fraction: {emo}"))

    for emo in PLUTCHIK:
        metrics.append((f"coverage_{emo}", f"Coverage: {emo}"))

    for emo in PLUTCHIK:
        metrics.append((f"corr_synt_{emo}", f"Corr syntactic-{emo}"))

    # Three-group comparison
    print("\n" + "=" * 100)
    print("  THREE-GROUP MULTIPLEX COMPARISON (length-matched)")
    print("=" * 100)

    all_results = []

    print("\n" + "-" * 100)
    all_results.extend(compare_groups("OCD", ocd, "Cancer", cancer, metrics))

    print("\n" + "-" * 100)
    all_results.extend(compare_groups("OCD", ocd, "Dep/PTSD", mh, metrics))

    # Summary of significant findings
    print("\n" + "=" * 100)
    print("  SIGNIFICANT MULTIPLEX FINDINGS")
    print("=" * 100)

    sig = [r for r in all_results if r["significant"]]
    if sig:
        for r in sorted(sig, key=lambda x: -abs(x["cohens_d"])):
            dir_ = "HIGHER" if r["cohens_d"] > 0 else "LOWER"
            print(f"  {r['comparison']:30s}: {r['metric']:40s} d={r['cohens_d']:+.3f} ({r['effect']}, p={r['p_value']:.6f})")
    else:
        print("  No significant multiplex differences found.")

    # Save comparison
    if all_results:
        path = os.path.join(HERE, "results", "multiplex_comparison.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\n  Saved to {path}")


if __name__ == "__main__":
    main()
