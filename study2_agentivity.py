"""
Study 2 — agentivity conditional on mention.

For every noun/pronoun slot directly attached to an Event (source of Agent->Event,
destination of Event->Target), record whether the word is the Agent or the Target.
Question: given that a category of words is mentioned, is it cast as Agent more
often in one disorder than in the others, beyond that group's overall agentivity?

Model per (category c, disorder g), aggregated binomial GLM with post-clustered SEs:
    logit P(Agent) = b0 + b1*is_c + b2*is_g + b3*is_c*is_g
exp(b3) is the agentivity odds ratio specific to category c in disorder g.
"""
import os
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm

from study2_reddit import load_merged_study2, bh_adjust, GRAPHS, RES
from study2_signatures import head, LABEL_WORDS, NS, GROUPS, SEED
from study2_agency import agent_category

OUT = os.path.join(RES, "signatures")


def slot_counts(slug):
    root = ET.parse(os.path.join(GRAPHS, f"{slug}.graphml")).getroot()
    keys = {k.get("id"): k.get("attr.name") for k in root.iter(f"{NS}key")}
    agent, target = Counter(), Counter()
    for e in root.iter(f"{NS}edge"):
        d = {keys[x.get("key")]: x.text for x in e.iter(f"{NS}data")}
        w = int(d.get("weight", 1))
        if d.get("tea_from") == "Agent" and d.get("tea_to") == "Event":
            h = head(e.get("source"))
            if h and h not in LABEL_WORDS:
                agent[agent_category(h)] += w
        elif d.get("tea_from") == "Event" and d.get("tea_to") == "Target":
            h = head(e.get("target"))
            if h and h not in LABEL_WORDS:
                target[agent_category(h)] += w
    return agent, target


def build_table(df):
    rows = []
    for i, (slug, g) in enumerate(zip(df["slug"], df["subreddit"])):
        a, t = slot_counts(slug)
        for c in set(a) | set(t):
            rows.append((slug, g, c, a[c], t[c]))
        if (i + 1) % 2500 == 0:
            print(f"  parsed {i+1}/{len(df)}", flush=True)
    return pd.DataFrame(rows, columns=["slug", "subreddit", "category", "agent", "target"])


def descriptives(tab, cats):
    tot = tab.groupby("subreddit")[["agent", "target"]].sum()
    base = (tot["agent"] / tot.sum(axis=1)).rename("all words")
    out = {"all words": base}
    mention = {}
    for c in cats:
        s = tab[tab.category == c].groupby("subreddit")[["agent", "target"]].sum()
        out[c] = s["agent"] / s.sum(axis=1)
        mention[c] = s.sum(axis=1) / tot.sum(axis=1)
    agentivity = pd.DataFrame(out).T[GROUPS] * 100
    mention = pd.DataFrame(mention).T[GROUPS] * 100
    return agentivity, mention


def fit_interaction(tab, c, g, posts):
    per_post = tab.groupby("slug")[["agent", "target"]].sum()
    cat = tab[tab.category == c].set_index("slug")[["agent", "target"]]
    cat = cat.reindex(per_post.index, fill_value=0)
    rest = per_post - cat
    is_g = (posts.loc[per_post.index, "subreddit"] == g).astype(float).to_numpy()
    rows = []
    for is_c, block in ((1.0, cat), (0.0, rest)):
        for slug, (a, t), ig in zip(block.index, block.to_numpy(), is_g):
            if a + t > 0:
                rows.append((slug, is_c, ig, a, t))
    d = pd.DataFrame(rows, columns=["slug", "is_c", "is_g", "agent", "target"])
    X = sm.add_constant(np.column_stack([d.is_c, d.is_g, d.is_c * d.is_g]))
    y = d[["agent", "target"]].to_numpy()
    groups = pd.factorize(d["slug"])[0]
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    b, se, p = fit.params[3], fit.bse[3], fit.pvalues[3]
    return b, se, p


def run_tests(tab, posts, cats):
    rows = []
    for c in cats:
        for g in GROUPS:
            b, se, p = fit_interaction(tab, c, g, posts)
            rows.append({"category": c, "group": g, "log_or": b, "OR": np.exp(b),
                         "ci_low": np.exp(b - 1.96 * se), "ci_high": np.exp(b + 1.96 * se), "p": p})
    res = pd.DataFrame(rows)
    res["p_bh"] = bh_adjust(res["p"])
    return res


def main():
    df = load_merged_study2()
    df = df[df["subreddit"].isin(GROUPS)].reset_index(drop=True)
    posts = df.set_index("slug")[["subreddit"]]
    print(f"Collecting role-tagged slots from {len(df)} graphs...")
    tab = build_table(df)
    tab.to_csv(os.path.join(OUT, "agentivity_slots.csv"), index=False)

    share = tab.groupby("category")[["agent", "target"]].sum().sum(axis=1)
    share = share / share.sum()
    cats = share[share >= 0.01].sort_values(ascending=False).index.tolist()
    cats = [c for c in cats if c != "not in WordNet"]

    agentivity, mention = descriptives(tab, cats)
    pd.set_option("display.width", 220)
    print("\nAgentivity = % of slots in which the word is the Agent (vs Target):")
    print(agentivity.round(1).to_string())
    print("\nMention rate = % of all slots occupied by the category:")
    print(mention.round(2).to_string())

    print("\nInteraction tests (category x disorder; OR > 1 = more agentive than expected)...")
    res = run_tests(tab, posts, cats)
    res.to_csv(os.path.join(OUT, "agentivity_interactions.csv"), index=False)
    piv = res.pivot(index="category", columns="group", values="OR")[GROUPS].loc[cats]
    sig = res.pivot(index="category", columns="group", values="p_bh")[GROUPS].loc[cats]
    fmt = piv.round(2).astype(str) + np.where(sig < .05, "*", " ")
    print(fmt.to_string())
    print("(* = p_BH < .05; BH over all category x group tests)")

    rng = np.random.default_rng(SEED)
    slugs = df["slug"].to_numpy()
    half = set(slugs[rng.random(len(slugs)) < 0.5])
    t1 = tab[tab.slug.isin(half)]
    t2 = tab[~tab.slug.isin(half)]
    r1 = run_tests(t1, posts, cats)
    r2 = run_tests(t2, posts, cats)
    m = r1[["category", "group", "OR"]].merge(r2[["category", "group", "OR"]], on=["category", "group"],
                                              suffixes=("_half1", "_half2"))
    m.to_csv(os.path.join(OUT, "agentivity_split_half.csv"), index=False)
    r = np.corrcoef(np.log(m.OR_half1), np.log(m.OR_half2))[0, 1]
    print(f"\nSplit-half: correlation of log ORs across halves r = {r:.3f}")
    key = res[res.p_bh < .05].merge(m, on=["category", "group"])
    print(key[["category", "group", "OR", "ci_low", "ci_high", "p_bh", "OR_half1", "OR_half2"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
