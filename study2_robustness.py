"""
Study 2 — robustness of the agentivity results to author non-independence.

Uses the passive-corrected slot table. Two checks on the category x disorder
interaction ORs:
  (1) cluster-robust SEs by author instead of by post ([deleted] = own cluster);
  (2) one randomly chosen post per author.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from study2_reddit import load_merged_study2, bh_adjust, RES, OUT as S2
from study2_signatures import GROUPS, SEED
from study2_passive import to_table, SLOTS

OUT = os.path.join(RES, "signatures")
KEY = ["self (I/we)", "noun.person", "they", "he/she", "noun.cognition",
       "absolutist pronoun", "noun.time", "noun.artifact", "noun.body"]


def fit(tab, posts, c, g):
    per_post = tab.groupby("slug")[["agent", "target"]].sum()
    cat = tab[tab.category == c].set_index("slug")[["agent", "target"]].reindex(per_post.index, fill_value=0)
    rest = per_post - cat
    meta = posts.loc[per_post.index]
    is_g = (meta["subreddit"] == g).to_numpy(float)
    clus = meta["cluster"].to_numpy()
    blocks = []
    for is_c, b in ((1.0, cat), (0.0, rest)):
        a, t = b["agent"].to_numpy(), b["target"].to_numpy()
        keep = (a + t) > 0
        blocks.append(pd.DataFrame({"is_c": is_c, "is_g": is_g[keep], "agent": a[keep],
                                    "target": t[keep], "cl": clus[keep]}))
    d = pd.concat(blocks, ignore_index=True)
    X = sm.add_constant(np.column_stack([d.is_c, d.is_g, d.is_c * d.is_g]))
    res = sm.GLM(d[["agent", "target"]].to_numpy(), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.cl)[0]})
    return res.params[3], res.bse[3], res.pvalues[3]


def run(tab, posts, label):
    rows = []
    for c in KEY:
        for g in GROUPS:
            b, se, p = fit(tab, posts, c, g)
            rows.append({"check": label, "category": c, "group": g, "OR": np.exp(b),
                         "ci_low": np.exp(b - 1.96 * se), "ci_high": np.exp(b + 1.96 * se), "p": p})
    r = pd.DataFrame(rows)
    r["p_bh"] = bh_adjust(r["p"])
    return r


def main():
    df = load_merged_study2()
    df = df[df["subreddit"].isin(GROUPS)].reset_index(drop=True)
    auth = pd.read_csv(os.path.join(S2, "metadata_authors.csv"))[["slug", "author"]]
    posts = df[["slug", "subreddit"]].merge(auth, on="slug", how="left").set_index("slug")
    deleted = posts["author"].isna() | (posts["author"] == "[deleted]")
    posts["cluster"] = np.where(deleted, "post:" + posts.index.to_series(), "author:" + posts["author"].astype(str))

    slots = pd.read_csv(SLOTS, keep_default_na=False, na_values=[""])
    slots = slots[slots["word"].notna() & slots.slug.isin(posts.index)]
    tab = to_table(slots, posts, corrected=True)

    post_cl = posts.assign(cluster="post:" + posts.index.to_series())
    r_post = run(tab, post_cl, "post clusters")
    r_auth = run(tab, posts, "author clusters")

    rng = np.random.default_rng(SEED)
    one = (posts.assign(_r=rng.random(len(posts))).sort_values("_r")
                .groupby("cluster").head(1).index)
    r_one = run(tab[tab.slug.isin(one)], posts, "one post per author")
    print(f"posts: {len(posts)}, clusters: {posts['cluster'].nunique()}, one-per-author sample: {len(one)}")

    allr = pd.concat([r_post, r_auth, r_one], ignore_index=True)
    allr.to_csv(os.path.join(OUT, "agentivity_robustness_authors.csv"), index=False)
    wide = allr.pivot_table(index=["category", "group"], columns="check", values="OR").reset_index()
    sig = allr.pivot_table(index=["category", "group"], columns="check", values="p_bh")
    wide = wide.merge(sig.add_prefix("pbh: ").reset_index(), on=["category", "group"])
    pd.set_option("display.width", 220)
    print(wide.round(3).to_string(index=False))
    ci = allr[allr.check == "author clusters"][["category", "group", "OR", "ci_low", "ci_high", "p_bh"]]
    ci.to_csv(os.path.join(OUT, "agentivity_final_author_clustered.csv"), index=False)


if __name__ == "__main__":
    main()
