"""
Study 2 — disorder signatures on Reddit (r/OCD, r/depression, r/ptsd, r/ADHD).

1. Structural profile: 9 TEA network metrics, residualised on log word count,
   Kruskal-Wallis + epsilon^2, one-vs-rest Hedges' g, BH-corrected.
2. Discriminability: multinomial logistic regression, 5-fold CV, balanced
   classes; structural metrics vs TEA content (node heads).
3. Content signatures: weighted log-odds with informative Dirichlet prior
   (Monroe et al., 2008) on Agent / Event / Target heads and Agent->Event pairs,
   counted as document frequency (one count per post).
"""
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from study2_reddit import load_merged_study2, bh_adjust, hedges_g, TEA_METRICS, RES, GRAPHS

OUT = os.path.join(RES, "signatures")
os.makedirs(OUT, exist_ok=True)
GROUPS = ["OCD", "depression", "ptsd", "ADHD"]
SEED = 42
# network structure only; thematic recurrence is analysed separately in study2_rqa_st.py
METRICS = TEA_METRICS

# disorder labels would trivially identify the subreddit
LABEL_WORDS = {"ocd", "adhd", "ptsd", "cptsd", "c-ptsd", "depression"}
NS = "{http://graphml.graphdrawing.org/xmlns}"


# ---------------------------------------------------------------- structure
def structural_profile(df):
    logw = np.log(df["n_words"].to_numpy())
    X = np.column_stack([np.ones_like(logw), logw])
    rows, resid = [], pd.DataFrame(index=df.index)
    for col, label in METRICS:
        y = df[col].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        resid[col] = r
        samples = [r[df["subreddit"] == g] for g in GROUPS]
        H, p_kw = stats.kruskal(*samples)
        n, k = len(r), len(GROUPS)
        eps2 = (H - k + 1) / (n - k)
        row = {"metric": label, "H": H, "p_kw": p_kw, "epsilon2": eps2}
        for g in GROUPS:
            mask = (df["subreddit"] == g).to_numpy()
            row[f"g_{g}"] = hedges_g(r[mask], r[~mask])
            row[f"p_{g}"] = stats.mannwhitneyu(r[mask], r[~mask]).pvalue
        rows.append(row)
    prof = pd.DataFrame(rows)
    prof["p_kw_bh"] = bh_adjust(prof["p_kw"])
    pcols = [f"p_{g}" for g in GROUPS]
    flat = bh_adjust(prof[pcols].to_numpy().ravel())
    prof[[f"pbh_{g}" for g in GROUPS]] = flat.reshape(-1, len(GROUPS))
    return prof, resid


# ------------------------------------------------------------------ content
def head(phrase):
    toks = re.findall(r"[a-z][a-z'\-]*", phrase.lower())
    return toks[-1] if toks else None


def read_graph_terms(slug):
    """Return sets of agent/event/target heads and agent->event head pairs for one post."""
    root = ET.parse(os.path.join(GRAPHS, f"{slug}.graphml")).getroot()
    keys = {k.get("id"): k.get("attr.name") for k in root.iter(f"{NS}key")}
    role = {}
    for node in root.iter(f"{NS}node"):
        for d in node.iter(f"{NS}data"):
            if keys.get(d.get("key")) == "tea_role":
                role[node.get("id")] = d.text
    terms = {"Agent": set(), "Event": set(), "Target": set(), "Agent->Event": set()}
    for nid, r in role.items():
        h = head(nid)
        if h and h not in LABEL_WORDS and r in terms:
            terms[r].add(h)
    for e in root.iter(f"{NS}edge"):
        s, t = e.get("source"), e.get("target")
        if role.get(s) == "Agent" and role.get(t) == "Event":
            hs, ht = head(s), head(t)
            if hs and ht and hs not in LABEL_WORDS and ht not in LABEL_WORDS:
                terms["Agent->Event"].add(f"{hs} -> {ht}")
    return terms


def weighted_log_odds(df_counts, n_docs, group, prior_scale=1000.0):
    """Monroe et al. (2008): group vs all other groups, informative Dirichlet prior."""
    y_i = df_counts[group]
    y_j = df_counts.drop(columns=group).sum(axis=1)
    pooled = df_counts.sum(axis=1)
    a_w = pooled / pooled.sum() * prior_scale
    a0 = a_w.sum()
    n_i = y_i.sum()
    n_j = y_j.sum()
    delta = (np.log((y_i + a_w) / (n_i + a0 - y_i - a_w))
             - np.log((y_j + a_w) / (n_j + a0 - y_j - a_w)))
    z = delta / np.sqrt(1 / (y_i + a_w) + 1 / (y_j + a_w))
    share = y_i / n_docs[group]
    others = n_docs.drop(group).sum()
    share_rest = y_j / others
    return pd.DataFrame({"z": z, "pct_group": 100 * share, "pct_rest": 100 * share_rest})


def content_signatures(df):
    docs = {}
    counts = {t: defaultdict(Counter) for t in ["Agent", "Event", "Target", "Agent->Event"]}
    for i, (slug, g) in enumerate(zip(df["slug"], df["subreddit"])):
        terms = read_graph_terms(slug)
        docs[slug] = terms
        for t, s in terms.items():
            counts[t][g].update(s)
        if (i + 1) % 2000 == 0:
            print(f"  parsed {i+1}/{len(df)} graphs", flush=True)
    n_docs = df["subreddit"].value_counts()

    tables = []
    for t, by_group in counts.items():
        mat = pd.DataFrame(by_group).fillna(0)[GROUPS]
        mat = mat[mat.sum(axis=1) >= 30]  # term must appear in >=30 posts overall
        for g in GROUPS:
            lo = weighted_log_odds(mat, n_docs, g)
            lo["role"], lo["group"] = t, g
            tables.append(lo.reset_index().rename(columns={"index": "term"}))
    return pd.concat(tables, ignore_index=True), docs


# ----------------------------------------------------------- discriminability
def balanced(df):
    n = df["subreddit"].value_counts().min()
    parts = [df[df["subreddit"] == g].sample(n, random_state=SEED) for g in GROUPS]
    return pd.concat(parts).reset_index(drop=True)


def evaluate(X, y, model, name):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    classes = np.unique(y)
    pred = classes[proba.argmax(axis=1)]
    res = {"features": name, "accuracy": accuracy_score(y, pred),
           "macro_auc": roc_auc_score(y, proba, multi_class="ovr", average="macro")}
    for k, c in enumerate(classes):
        res[f"auc_{c}"] = roc_auc_score(y == c, proba[:, k])
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)
    return res, cm


def discriminability(df, resid, docs):
    bal = balanced(df.assign(_idx=df.index))
    y = bal["subreddit"].to_numpy()

    X_struct = resid.loc[bal["_idx"], [c for c, _ in METRICS]].to_numpy()
    m_struct = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    r1, cm1 = evaluate(X_struct, y, m_struct, "structural (9 TEA network metrics)")

    def doc_string(slug):
        t = docs[slug]
        return " ".join([f"A_{w}" for w in t["Agent"]] + [f"E_{w}" for w in t["Event"]]
                        + [f"T_{w}" for w in t["Target"]])
    texts = [doc_string(s) for s in bal["slug"]]
    m_content = make_pipeline(TfidfVectorizer(min_df=5, token_pattern=r"\S+", sublinear_tf=True),
                              LogisticRegression(max_iter=2000, C=1.0))
    r2, cm2 = evaluate(np.array(texts, dtype=object), y, m_content, "TEA content (A/E/T heads)")
    return pd.DataFrame([r1, r2]), cm1, cm2, len(bal)


# --------------------------------------------------------------------- main
def main():
    df = load_merged_study2()
    df = df[df["subreddit"].isin(GROUPS)].reset_index(drop=True)
    print(df["subreddit"].value_counts().to_string(), "\n")

    prof, resid = structural_profile(df)
    prof.to_csv(os.path.join(OUT, "structural_profile.csv"), index=False)
    pd.set_option("display.width", 220)
    show = prof[["metric", "epsilon2", "p_kw_bh"] + [f"g_{g}" for g in GROUPS]].copy()
    print("=== Structural profile (residualised on log length; g = group vs other three) ===")
    print(show.round(3).to_string(index=False))

    print("\nParsing TEA graphs for content signatures...")
    sig, docs = content_signatures(df)
    sig.to_csv(os.path.join(OUT, "content_log_odds.csv"), index=False)
    for g in GROUPS:
        print(f"\n=== r/{g}: most distinctive TEA terms (z > 0, top 12 per role) ===")
        for role in ["Agent", "Event", "Target", "Agent->Event"]:
            top = sig[(sig.group == g) & (sig.role == role)].nlargest(12, "z")
            items = ", ".join(f"{t} ({z:.1f})" for t, z in zip(top["term"], top["z"]))
            print(f"  {role:13s} {items}")

    print("\n=== Discriminability (5-fold CV, balanced classes) ===")
    disc, cm1, cm2, n_bal = discriminability(df, resid, docs)
    disc.to_csv(os.path.join(OUT, "discriminability.csv"), index=False)
    print(f"n = {n_bal} posts ({n_bal // 4} per group); chance accuracy = 0.25, chance AUC = 0.50")
    print(disc.round(3).to_string(index=False))
    print("\nConfusion (structural):\n", cm1.to_string())
    print("\nConfusion (content):\n", cm2.to_string())


if __name__ == "__main__":
    main()
