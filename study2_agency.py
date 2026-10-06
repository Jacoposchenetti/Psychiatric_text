"""
Study 2 — does TEA role information matter?

A. Ablation: classifiers on raw words vs TEA heads without roles vs with roles,
   single slots, and raw words + TEA roles. Same balanced sample and CV folds;
   McNemar tests on paired predictions.
B. Agency profile: each Agent slot classified with an external taxonomy
   (closed pronoun classes, then WordNet first-sense noun supersense),
   per-post weighted shares compared across disorders (one-vs-rest Hedges' g,
   Kruskal-Wallis epsilon^2, BH), with a split-half stability check.
"""
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from study2_reddit import load_merged_study2, bh_adjust, hedges_g, GRAPHS, TEXTS, RES
from study2_signatures import head, LABEL_WORDS, NS, GROUPS, balanced, SEED

OUT = os.path.join(RES, "signatures")

PRONOUN_CLASSES = {
    "self (I/we)": {"i", "me", "myself", "we", "us", "ourselves", "my"},
    "you": {"you", "yourself", "yourselves"},
    "he/she": {"he", "she", "him", "her", "himself", "herself"},
    "they": {"they", "them", "themselves"},
    "it/this/that": {"it", "this", "that", "these", "those", "which", "what", "there", "itself"},
    "absolutist pronoun": {"everyone", "everybody", "everything", "nobody", "nothing",
                           "none", "all", "noone"},
    "other indefinite": {"someone", "somebody", "something", "anyone", "anybody",
                         "anything", "one"},
}
PRONOUN_OF = {w: c for c, ws in PRONOUN_CLASSES.items() for w in ws}


def agent_category(word, cache={}):
    if word in PRONOUN_OF:
        return PRONOUN_OF[word]
    if word not in cache:
        base = wn.morphy(word, wn.NOUN) or word
        syn = wn.synsets(base, pos=wn.NOUN)
        cache[word] = syn[0].lexname() if syn else "not in WordNet"
    return cache[word]


def parse_graph(slug):
    root = ET.parse(os.path.join(GRAPHS, f"{slug}.graphml")).getroot()
    keys = {k.get("id"): k.get("attr.name") for k in root.iter(f"{NS}key")}
    role = {}
    for node in root.iter(f"{NS}node"):
        for d in node.iter(f"{NS}data"):
            if keys.get(d.get("key")) == "tea_role":
                role[node.get("id")] = d.text
    sets = {"Agent": set(), "Event": set(), "Target": set()}
    for nid, r in role.items():
        h = head(nid)
        if h and h not in LABEL_WORDS and r in sets:
            sets[r].add(h)
    agent_w = Counter()  # agent head -> number of Agent->Event triplets
    for e in root.iter(f"{NS}edge"):
        s = e.get("source")
        if role.get(s) != "Agent":
            continue
        w = 1
        for d in e.iter(f"{NS}data"):
            if keys.get(d.get("key")) == "weight":
                w = int(d.text)
        h = head(s)
        if h and h not in LABEL_WORDS:
            agent_w[h] += w
    return sets, agent_w


def raw_text(slug):
    with open(os.path.join(TEXTS, f"{slug}.txt"), encoding="utf-8") as f:
        toks = re.findall(r"[a-z][a-z'\-]*", f.read().lower())
    return " ".join(t for t in toks if t not in LABEL_WORDS)


# ---------------------------------------------------------------- ablation
def cv_predict(texts, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    model = make_pipeline(TfidfVectorizer(min_df=5, token_pattern=r"\S+", sublinear_tf=True),
                          LogisticRegression(max_iter=3000))
    proba = cross_val_predict(model, np.array(texts, dtype=object), y, cv=cv, method="predict_proba")
    classes = np.unique(y)
    pred = classes[proba.argmax(axis=1)]
    return pred, proba, classes


def mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def ablation(bal, parsed):
    y = bal["subreddit"].to_numpy()
    tag = lambda s, r: [f"{r[0]}_{w}" for w in parsed[s][0][r]]
    feats = {
        "raw words (full post)": [raw_text(s) for s in bal["slug"]],
        "TEA heads, no roles": [" ".join(parsed[s][0]["Agent"] | parsed[s][0]["Event"] | parsed[s][0]["Target"])
                                for s in bal["slug"]],
        "TEA heads + roles": [" ".join(tag(s, "Agent") + tag(s, "Event") + tag(s, "Target")) for s in bal["slug"]],
        "Agent slot only": [" ".join(tag(s, "Agent")) for s in bal["slug"]],
        "Event slot only": [" ".join(tag(s, "Event")) for s in bal["slug"]],
        "Target slot only": [" ".join(tag(s, "Target")) for s in bal["slug"]],
    }
    feats["raw words + TEA roles"] = [w + " " + r for w, r in zip(feats["raw words (full post)"],
                                                                  feats["TEA heads + roles"])]
    rows, correct = [], {}
    for name, texts in feats.items():
        pred, proba, classes = cv_predict(texts, y)
        correct[name] = pred == y
        row = {"features": name, "accuracy": accuracy_score(y, pred),
               "macro_auc": roc_auc_score(y, proba, multi_class="ovr", average="macro")}
        for k, c in enumerate(classes):
            row[f"auc_{c}"] = roc_auc_score(y == c, proba[:, k])
        rows.append(row)
        print(f"  {name:28s} acc={row['accuracy']:.3f}  macroAUC={row['macro_auc']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    tests = []
    for a, b in [("TEA heads + roles", "TEA heads, no roles"),
                 ("raw words + TEA roles", "raw words (full post)"),
                 ("TEA heads + roles", "raw words (full post)")]:
        only_a, only_b, p = mcnemar(correct[a], correct[b])
        tests.append({"A": a, "B": b, "acc_A": correct[a].mean(), "acc_B": correct[b].mean(),
                      "A_right_B_wrong": only_a, "B_right_A_wrong": only_b, "p_mcnemar": p})
    return res, pd.DataFrame(tests)


# ------------------------------------------------------------------ agency
def agency_shares(df, parsed, min_slots=5):
    rows = []
    for slug, g in zip(df["slug"], df["subreddit"]):
        aw = parsed[slug][1]
        total = sum(aw.values())
        if total < min_slots:
            continue
        cat = Counter()
        for w, n in aw.items():
            cat[agent_category(w)] += n
        row = {k: v / total for k, v in cat.items()}
        row.update(slug=slug, subreddit=g, n_agent_slots=total)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def agency_tests(shares, min_mean=0.01):
    cats = [c for c in shares.columns if c not in ("slug", "subreddit", "n_agent_slots")]
    cats = [c for c in cats if shares[c].mean() >= min_mean]
    rows = []
    for c in cats:
        x = shares[c].to_numpy()
        samples = [x[shares["subreddit"] == g] for g in GROUPS]
        H, p = stats.kruskal(*samples)
        row = {"category": c, "overall_pct": 100 * x.mean(),
               "epsilon2": (H - len(GROUPS) + 1) / (len(x) - len(GROUPS)), "p_kw": p}
        for g in GROUPS:
            m = (shares["subreddit"] == g).to_numpy()
            row[f"pct_{g}"] = 100 * x[m].mean()
            row[f"g_{g}"] = hedges_g(x[m], x[~m])
            row[f"p_{g}"] = stats.mannwhitneyu(x[m], x[~m]).pvalue
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("overall_pct", ascending=False).reset_index(drop=True)
    t["p_kw_bh"] = bh_adjust(t["p_kw"])
    flat = bh_adjust(t[[f"p_{g}" for g in GROUPS]].to_numpy().ravel())
    t[[f"pbh_{g}" for g in GROUPS]] = flat.reshape(-1, len(GROUPS))
    return t


def split_half(shares, cats):
    rng = np.random.default_rng(SEED)
    half = rng.random(len(shares)) < 0.5
    out = []
    for c in cats:
        for g in GROUPS:
            gs = []
            for h in (half, ~half):
                s = shares[h]
                m = (s["subreddit"] == g).to_numpy()
                gs.append(hedges_g(s[c].to_numpy()[m], s[c].to_numpy()[~m]))
            out.append({"category": c, "group": g, "g_half1": gs[0], "g_half2": gs[1]})
    return pd.DataFrame(out)


def main():
    df = load_merged_study2()
    df = df[df["subreddit"].isin(GROUPS)].reset_index(drop=True)
    print(f"Parsing {len(df)} TEA graphs...", flush=True)
    parsed = {s: parse_graph(s) for s in df["slug"]}

    print("\n=== A. Ablation (balanced, 5-fold CV, same folds) ===")
    bal = balanced(df)
    print(f"n = {len(bal)} ({len(bal)//4} per group)")
    abl, tests = ablation(bal, parsed)
    abl.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    tests.to_csv(os.path.join(OUT, "ablation_mcnemar.csv"), index=False)
    print("\nMcNemar (paired predictions):")
    print(tests.round(4).to_string(index=False))

    print("\n=== B. Agency profile (share of Agent slots per post) ===")
    shares = agency_shares(df, parsed)
    print(f"posts with >=5 agent slots: {len(shares)}  "
          f"(median slots/post = {shares['n_agent_slots'].median():.0f})")
    t = agency_tests(shares)
    t.to_csv(os.path.join(OUT, "agency_profile.csv"), index=False)
    pd.set_option("display.width", 230)
    cols = ["category", "overall_pct", "epsilon2"] + [f"pct_{g}" for g in GROUPS] + [f"g_{g}" for g in GROUPS]
    print(t[cols].round(3).to_string(index=False))

    sh = split_half(shares, t["category"].tolist())
    sh.to_csv(os.path.join(OUT, "agency_split_half.csv"), index=False)
    big = sh[(sh[["g_half1", "g_half2"]].abs().max(axis=1) >= 0.2)]
    print("\nSplit-half stability for effects with |g| >= 0.2 in either half:")
    print(big.round(3).to_string(index=False))
    r = np.corrcoef(sh["g_half1"], sh["g_half2"])[0, 1]
    print(f"Correlation of g across halves (all category x group cells): r = {r:.3f}")

    top = Counter()
    for s in shares["slug"]:
        top.update(parsed[s][1])
    ex = {}
    for w, n in top.most_common(400):
        ex.setdefault(agent_category(w), []).append(w)
    print("\nMost frequent agent heads per category (for transparency):")
    for c in t["category"]:
        print(f"  {c:22s} {', '.join(ex.get(c, [])[:10])}")


if __name__ == "__main__":
    main()
