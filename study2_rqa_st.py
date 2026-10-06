"""
Study 2 — thematic recurrence with sentence-transformer embeddings.

Sentences are segmented once with spaCy's statistical sentence segmenter and
embedded with all-MiniLM-L6-v2 (main analysis) and all-mpnet-base-v2 (check).
Averaged word vectors are not used: they are dominated by function words and
measure style as much as content. Four recurrence thresholds: global thresholds
giving a pooled recurrence rate of 5%, 10% or 15%, and a per-text threshold
fixing each post's recurrence rate at 10%.

Usage: python study2_rqa_st.py [segment|embed MODEL|rqa|analyse]
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from study2_reddit import load_merged_study2, bh_adjust, hedges_g, TEXTS, OUT as S2
from study2_signatures import GROUPS, SEED

D = os.path.join(S2, "rqa_st")
os.makedirs(D, exist_ok=True)
SENTS = os.path.join(D, "sentences.json")
MODELS = {"minilm": "all-MiniLM-L6-v2", "mpnet": "all-mpnet-base-v2"}
SCHEMES = {"global05": 0.05, "global10": 0.10, "global15": 0.15, "pertext10": 0.10}
METRICS = ["RR", "DET", "LAM", "TT", "L", "Lmax", "ENTR"]
MIN_SENTS = 10


# ------------------------------------------------------------------ segment
def segment():
    import spacy
    df = load_merged_study2()
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "lemmatizer", "attribute_ruler", "tagger"])
    nlp.enable_pipe("senter")
    texts = [open(os.path.join(TEXTS, f"{s}.txt"), encoding="utf-8").read()[:45000] for s in df["slug"]]
    out, t0 = {}, time.time()
    for i, (doc, slug) in enumerate(zip(nlp.pipe(texts, batch_size=64), df["slug"])):
        # vector_norm > 0 drops sentences made only of out-of-vocabulary tokens (URL debris, symbols)
        out[slug] = [s.text.strip() for s in doc.sents if len(s.text.split()) >= 3 and s.vector_norm > 0]
        if (i + 1) % 2000 == 0:
            print(f"[segment {i+1}/{len(df)}] {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(SENTS, "w", encoding="utf-8"))
    print(f"{sum(map(len, out.values()))} sentences from {len(out)} posts")


def sentence_index():
    sents = json.load(open(SENTS, encoding="utf-8"))
    offsets, pos = {}, 0
    for slug, s in sents.items():
        offsets[slug] = (pos, pos + len(s))
        pos += len(s)
    return sents, offsets


# -------------------------------------------------------------------- embed
def embed(key):
    from sentence_transformers import SentenceTransformer
    sents, _ = sentence_index()
    flat = [s for v in sents.values() for s in v]
    model = SentenceTransformer(MODELS[key], device="cpu")
    t0 = time.time()
    emb = model.encode(flat, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    np.save(os.path.join(D, f"emb_{key}.npy"), emb.astype(np.float16))
    print(f"{key}: {len(flat)} sentences in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------- rqa
def runs(binary_rows):
    """Lengths of runs of 1s along each row of a 2-D 0/1 array."""
    z = np.zeros((binary_rows.shape[0], 1), dtype=np.int8)
    d = np.diff(np.hstack([z, binary_rows.astype(np.int8), z]), axis=1)
    return np.nonzero(d == -1)[1] - np.nonzero(d == 1)[1]


def rqa(R):
    """Same definitions as rqa_three_groups.compute_rqa, vectorised (lines of length >= 2)."""
    n = R.shape[0]
    total = R.sum()
    if total == 0:
        return None
    diag = []
    for k in range(1, n):
        for sign in (1, -1):
            a = np.diag(R, sign * k)
            lens = runs(a[None, :])
            diag.append(lens[lens >= 2])
    diag = np.concatenate(diag) if diag else np.array([], dtype=int)
    vert = runs(R.T)
    vert = vert[vert >= 2]
    if len(diag):
        _, counts = np.unique(diag, return_counts=True)
        p = counts / counts.sum()
        entr = float(-(p * np.log2(p)).sum())
    else:
        entr = 0.0
    return {"n_sentences": n, "RR": total / (n * (n - 1)),
            "DET": diag.sum() / total, "L": diag.mean() if len(diag) else 0.0,
            "Lmax": int(diag.max()) if len(diag) else 0, "ENTR": entr,
            "LAM": vert.sum() / total, "TT": vert.mean() if len(vert) else 0.0}


def similarity(E):
    E = E.astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    return S


def run_rqa():
    sents, offsets = sentence_index()
    slugs = [s for s, (a, b) in offsets.items() if b - a >= MIN_SENTS]
    rng = np.random.default_rng(SEED)
    calib = rng.choice(slugs, size=500, replace=False)
    rows = []
    for key in MODELS:
        path = os.path.join(D, f"emb_{key}.npy")
        if not os.path.exists(path):
            print(f"skip {key}: no embeddings")
            continue
        emb = np.load(path, mmap_mode="r")
        pooled = []
        for s in calib:
            a, b = offsets[s]
            S = similarity(np.array(emb[a:b]))
            pooled.append(S[np.triu_indices_from(S, k=1)])
        pooled = np.concatenate(pooled)
        thr = {k: float(np.quantile(pooled, 1 - rr)) for k, rr in SCHEMES.items() if k.startswith("global")}
        print(f"{key}: global thresholds {thr}", flush=True)
        t0 = time.time()
        for i, s in enumerate(slugs):
            a, b = offsets[s]
            S = similarity(np.array(emb[a:b]))
            upper = S[np.triu_indices_from(S, k=1)]
            for scheme, rr in SCHEMES.items():
                t = thr[scheme] if scheme in thr else float(np.quantile(upper, 1 - rr))
                res = rqa((S >= t).astype(np.int8))
                if res is not None:
                    res.update(slug=s, embedding=key, scheme=scheme, threshold=t)
                    rows.append(res)
            if (i + 1) % 2500 == 0:
                print(f"  [{key} {i+1}/{len(slugs)}] {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(D, "rqa_st_metrics.csv"), index=False)
    print(f"saved {len(rows)} rows")


# ------------------------------------------------------------------ analyse
def analyse():
    m = pd.read_csv(os.path.join(D, "rqa_st_metrics.csv"))
    meta = load_merged_study2()[["slug", "subreddit", "n_words"]]
    m = m.merge(meta, on="slug")
    rows = []
    for (emb, scheme), g in m.groupby(["embedding", "scheme"]):
        logn = np.log(g["n_sentences"].to_numpy(float))
        X = np.column_stack([np.ones_like(logn), logn])
        for met in METRICS:
            if scheme == "pertext10" and met == "RR":
                continue  # fixed by construction
            y = g[met].to_numpy(float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            groups = g["subreddit"].to_numpy()
            H, p = stats.kruskal(*[r[groups == k] for k in GROUPS])
            row = {"embedding": emb, "scheme": scheme, "metric": met, "n": len(r),
                   "epsilon2": (H - 3) / (len(r) - 4), "p_kw": p}
            for k in GROUPS:
                mask = groups == k
                row[f"g_{k}"] = hedges_g(r[mask], r[~mask])
                row[f"mean_{k}"] = g.loc[mask, met].mean()
            rows.append(row)
    res = pd.DataFrame(rows)
    res["p_bh"] = bh_adjust(res["p_kw"])
    res.to_csv(os.path.join(D, "rqa_st_profile.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["embedding", "scheme", "metric", "epsilon2"] + [f"g_{k}" for k in GROUPS]
    order = {"minilm": 0, "mpnet": 1}
    res = res.sort_values(["scheme", "metric", "embedding"], key=lambda s: s.map(order) if s.name == "embedding" else s)
    for scheme in SCHEMES:
        print(f"\n=== {scheme} (metrics residualised on log #sentences; g = community vs other three) ===")
        print(res[res.scheme == scheme][cols].round(3).to_string(index=False))

    print("\n=== Largest effects (|g| >= 0.2) ===")
    long = res.melt(id_vars=["embedding", "scheme", "metric"], value_vars=[f"g_{k}" for k in GROUPS],
                    var_name="group", value_name="g")
    print(long[long.g.abs() >= 0.2].sort_values("g").round(3).to_string(index=False))

    print("\n=== Agreement between embeddings (correlation of the g profiles across scheme x metric x group) ===")
    wide = long.pivot_table(index=["scheme", "metric", "group"], columns="embedding", values="g")
    print(wide.corr().round(3))

    g10 = m[m.scheme == "global10"].pivot_table(index="slug", columns="embedding", values=["RR", "DET", "LAM"])
    print("\n=== Per-post correlation between embeddings (global10) ===")
    for met in ["RR", "DET", "LAM"]:
        print(met, g10[met].corr(method="spearman").round(3).to_dict())

    # how much recurrence alone tells the communities apart
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    print("\n=== Classification from the 7 recurrence metrics + log #sentences (balanced, 5-fold) ===")
    for emb in m.embedding.unique():
        d = m[(m.embedding == emb) & (m.scheme == "global10")].copy()
        nmin = d.subreddit.value_counts().min()
        d = pd.concat([d[d.subreddit == k].sample(nmin, random_state=SEED) for k in GROUPS])
        X = np.column_stack([d[METRICS].to_numpy(float), np.log(d["n_sentences"])])
        y = d["subreddit"].to_numpy()
        proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")
        classes = np.unique(y)
        acc = (classes[proba.argmax(1)] == y).mean()
        print(f"  {emb:7s} accuracy={acc:.3f} macroAUC={roc_auc_score(y, proba, multi_class='ovr'):.3f}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "segment":
        segment()
    elif stage == "embed":
        embed(sys.argv[2])
    elif stage == "rqa":
        run_rqa()
    elif stage == "analyse":
        analyse()
    else:
        print(__doc__)
