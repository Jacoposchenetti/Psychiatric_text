"""
Study 2: within-platform replication on Reddit.
r/OCD vs r/depression, r/ptsd, r/ADHD (separate length-matched comparisons),
plus a platform check: OCD Stories (blog) vs r/OCD.

Stages (each cached on disk):  build -> tea -> rqa -> compare
Usage: python study2_reddit.py [build|tea|rqa|compare|all]
"""
import os
import sys
import csv
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "study2_reddit")
TEXTS = os.path.join(OUT, "texts")
RES = os.path.join(OUT, "results")
GRAPHS = os.path.join(RES, "graphs")
for d in (TEXTS, RES, GRAPHS):
    os.makedirs(d, exist_ok=True)

SUBREDDITS = ["OCD", "depression", "ptsd", "ADHD"]
N_PER_GROUP = None  # None = every eligible post
MIN_WORDS = 300
SEED = 42
RQA_THRESHOLD = 0.923  # same threshold as Study 1, so RQA values are on the same scale
MATCH_TOL = 0.35

TEA_METRICS = [
    ("first_person_ratio", "First-Person Agent Ratio"),
    ("lscc_frac", "LSCC Fraction"),
    ("triplets_per_word", "TEA Triplets/Word"),
    ("density", "Graph Density"),
    ("avg_clustering", "Avg Clustering"),
    ("node_ttr", "Node Type-Token Ratio"),
    ("repetition_idx", "Edge Repetition Index"),
    ("mean_edge_weight", "Mean Edge Weight"),
    ("nodes_per_word", "Unique Nodes/Word"),
]
RQA_METRICS = [
    ("RR", "Recurrence Rate"),
    ("DET", "Determinism"),
    ("LAM", "Laminarity"),
    ("TT", "Trapping Time"),
    ("L", "Mean Diagonal L"),
    ("Lmax", "Max Diagonal Lmax"),
    ("ENTR", "Recurrence Entropy"),
]


# --------------------------------------------------------------------- build
def build():
    from datasets import load_dataset
    from build_reddit_mh_control import clean_reddit_text

    df = pd.DataFrame(load_dataset("solomonk/reddit_mental_health_posts", split="train"))
    df = df[df["subreddit"].isin(SUBREDDITS)].copy()
    df["body"] = df["body"].fillna("")
    df = df[~df["body"].str.strip().isin(["[removed]", "[deleted]", ""])]
    df["clean_text"] = df["body"].apply(clean_reddit_text)
    df["wc"] = df["clean_text"].str.split().str.len()
    df = df[df["wc"] >= MIN_WORDS]
    df = df.drop_duplicates(subset="clean_text")

    # keep Study 2 independent of the Study 1 Reddit control sample
    s1 = pd.read_csv(os.path.join(HERE, "control_mental_health", "metadata.csv"))
    s1_ids = set(s1["slug"].str.split("_", n=1).str[1])
    df = df[~df["id"].isin(s1_ids)]

    rows = []
    for sub in SUBREDDITS:
        pool = df[df["subreddit"] == sub]
        n = len(pool) if N_PER_GROUP is None else min(N_PER_GROUP, len(pool))
        sample = pool.sample(n, random_state=SEED)
        print(f"r/{sub}: pool={len(pool)}, sampled={len(sample)}, "
              f"words M={sample['wc'].mean():.0f} SD={sample['wc'].std():.0f}")
        for _, r in sample.iterrows():
            slug = f"{sub}_{r['id']}"
            text = r["clean_text"]
            title = str(r.get("title", "")).strip()
            if title and title.lower() != "nan":
                text = title + "\n\n" + text
            with open(os.path.join(TEXTS, f"{slug}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            rows.append({"slug": slug, "subreddit": sub, "word_count": len(text.split())})

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "metadata.csv"), index=False)
    print(f"Saved {len(rows)} texts to {TEXTS}")


def read_text(slug):
    with open(os.path.join(TEXTS, f"{slug}.txt"), encoding="utf-8") as f:
        return f.read()


# ----------------------------------------------------------------------- tea
def tea():
    import networkx as nx
    import spacy
    sys.path.insert(0, r"C:\Users\jsche\Desktop\NLP\TEA_Networks")
    from tea_mh_control_analysis import extract_tea_for_text, build_tea_graph, compute_metrics

    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    meta = pd.read_csv(os.path.join(OUT, "metadata.csv"))
    out_csv = os.path.join(RES, "tea_metrics.csv")
    done = set(pd.read_csv(out_csv)["slug"]) if os.path.exists(out_csv) else set()

    t0 = time.time()
    for i, r in enumerate(meta.itertuples()):
        if r.slug in done:
            continue
        text = read_text(r.slug)
        df = extract_tea_for_text(text, nlp)
        G = build_tea_graph(df)
        m = compute_metrics(G, df, text)
        if m is None:
            continue
        m["slug"], m["subreddit"] = r.slug, r.subreddit
        nx.write_graphml(G, os.path.join(GRAPHS, f"{r.slug}.graphml"))
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(m.keys()))
            if write_header:
                w.writeheader()
            w.writerow(m)
        if (i + 1) % 250 == 0:
            print(f"[TEA {i+1}/{len(meta)}] {time.time()-t0:.0f}s", flush=True)
    print(f"TEA done in {time.time()-t0:.0f}s")


# ----------------------------------------------------------------------- rqa
def rqa():
    import spacy
    from rqa_three_groups import text_to_sentence_embeddings, build_recurrence_matrix, compute_rqa

    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    meta = pd.read_csv(os.path.join(OUT, "metadata.csv"))
    rows = []
    t0 = time.time()
    for i, r in enumerate(meta.itertuples()):
        text = read_text(r.slug)
        vecs = text_to_sentence_embeddings(text, nlp)
        if vecs is None or len(vecs) < 10:
            continue
        res = compute_rqa(build_recurrence_matrix(vecs, RQA_THRESHOLD))
        if res is None:
            continue
        res.update(slug=r.slug, subreddit=r.subreddit, n_words=len(text.split()))
        rows.append(res)
        if (i + 1) % 500 == 0:
            print(f"[RQA {i+1}/{len(meta)}] {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "rqa_metrics.csv"), index=False)
    print(f"RQA done: {len(rows)} texts in {time.time()-t0:.0f}s")


# ------------------------------------------------------------------- compare
def hedges_g(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    d = (np.mean(a) - np.mean(b)) / pooled
    return d * (1 - 3 / (4 * (na + nb) - 9))


def match_by_word_count(a, b, tol=MATCH_TOL, seed=SEED):
    """Greedy 1:1 nearest-neighbour matching on n_words (same rule as Study 1)."""
    a = a.sample(frac=1, random_state=seed).reset_index(drop=True)
    b = b.reset_index(drop=True)
    b_wc = b["n_words"].to_numpy(dtype=float)
    used = np.zeros(len(b), dtype=bool)
    ia_keep, ib_keep = [], []
    for ia, wc in enumerate(a["n_words"].to_numpy(dtype=float)):
        diffs = np.where(used, np.inf, np.abs(b_wc - wc))
        ib = int(np.argmin(diffs))
        if np.isfinite(diffs[ib]) and diffs[ib] / max(wc, 1) <= tol:
            used[ib] = True
            ia_keep.append(ia)
            ib_keep.append(ib)
    return a.iloc[ia_keep].reset_index(drop=True), b.iloc[ib_keep].reset_index(drop=True)


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def load_merged_study2():
    t = pd.read_csv(os.path.join(RES, "tea_metrics.csv"))
    q = pd.read_csv(os.path.join(RES, "rqa_metrics.csv"))
    t["triplets_per_word"] = t["n_triplets"] / t["n_words"]
    t["nodes_per_word"] = t["n_nodes"] / t["n_words"]
    q = q.drop(columns=["subreddit", "n_words"])
    return t.merge(q, on="slug", how="inner")


def load_merged_ocd_stories():
    t = pd.read_csv(os.path.join(HERE, "results", "tea_metrics.csv"))
    q = pd.read_csv(os.path.join(HERE, "results", "rqa_ocd.csv"))
    t["triplets_per_word"] = t["n_triplets"] / t["n_words"]
    t["nodes_per_word"] = t["n_nodes"] / t["n_words"]
    q = q.drop(columns=[c for c in ["n_words", "n_sentences"] if c in q.columns])
    m = t.merge(q, on="slug", how="inner")
    m["subreddit"] = "OCD Stories (blog)"
    return m


def compare_pair(a, b, name_a, name_b, family):
    am, bm = match_by_word_count(a, b)
    _, p_wc = stats.mannwhitneyu(am["n_words"], bm["n_words"])
    print(f"\n{name_a} vs {name_b}: {len(am)} matched pairs "
          f"(words {am['n_words'].mean():.0f} vs {bm['n_words'].mean():.0f}, "
          f"g_wc={hedges_g(am['n_words'], bm['n_words']):+.3f}, p_wc={p_wc:.3f})")
    rows = []
    for kind, metrics in (("TEA", TEA_METRICS), ("RQA", RQA_METRICS)):
        for col, label in metrics:
            va, vb = am[col].dropna().to_numpy(), bm[col].dropna().to_numpy()
            _, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
            rows.append({
                "family": family, "comparison": f"{name_a} vs {name_b}", "method": kind,
                "metric": label, "n_pairs": len(am),
                "mean_a": va.mean(), "sd_a": va.std(ddof=1),
                "mean_b": vb.mean(), "sd_b": vb.std(ddof=1),
                "hedges_g": hedges_g(va, vb), "p": p,
            })
    return rows


def compare():
    s2 = load_merged_study2()
    print("Study 2 texts with both TEA and RQA:")
    print(s2.groupby("subreddit")["n_words"].agg(["count", "mean", "std", "median"]).round(0))

    desc = s2.groupby("subreddit")[[c for c, _ in TEA_METRICS + RQA_METRICS]].mean().T
    desc.to_csv(os.path.join(RES, "group_means_unmatched.csv"))
    print("\nUnmatched group means (key metrics):")
    print(desc.loc[["first_person_ratio", "RR", "DET", "LAM", "ENTR"]].round(3))

    ocd = s2[s2["subreddit"] == "OCD"]
    rows = []
    for ctrl in ["depression", "ptsd", "ADHD"]:
        rows += compare_pair(ocd, s2[s2["subreddit"] == ctrl], "r/OCD", f"r/{ctrl}", "within-reddit")

    res = pd.DataFrame(rows)
    res["p_bh"] = bh_adjust(res["p"])  # BH over all within-Reddit tests

    blog = load_merged_ocd_stories()
    plat = pd.DataFrame(compare_pair(blog, ocd, "OCD Stories", "r/OCD", "platform"))
    plat["p_bh"] = bh_adjust(plat["p"])  # separate family: platform check

    out = pd.concat([res, plat], ignore_index=True)
    out.to_csv(os.path.join(RES, "study2_comparisons.csv"), index=False)

    pd.set_option("display.width", 200)
    for comp, g in out.groupby("comparison", sort=False):
        print(f"\n=== {comp} (n={g['n_pairs'].iloc[0]} pairs) ===")
        show = g[["method", "metric", "mean_a", "mean_b", "hedges_g", "p", "p_bh"]].copy()
        show["sig"] = np.where(show["p_bh"] < .05, "*", "")
        print(show.round({"mean_a": 3, "mean_b": 3, "hedges_g": 2, "p": 4, "p_bh": 4}).to_string(index=False))

    wr = out[out["family"] == "within-reddit"]
    print(f"\nWithin-Reddit: {(wr['p'] < .05).sum()}/{len(wr)} uncorrected, "
          f"{(wr['p_bh'] < .05).sum()}/{len(wr)} after BH")
    print(f"Saved {os.path.join(RES, 'study2_comparisons.csv')}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("build", "all"):
        build()
    if stage in ("tea", "all"):
        tea()
    if stage in ("rqa", "all"):
        rqa()
    if stage in ("compare", "all"):
        compare()
