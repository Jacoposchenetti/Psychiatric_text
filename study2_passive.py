"""
Study 2 — agentivity with passive-voice correction.

TEA places the patient of an agentless passive ("I was assaulted") in the Agent
slot and flags the row with passive_approx=1 (passives with a by-phrase are
already resolved: by-agent -> Agent, patient -> Target). The saved graphml files
lost that flag, so triplets are re-extracted here and the agentivity analysis is
run twice on the same rows: as extracted, and with passive_approx subjects
moved to the Target (patient) role.

Usage: python study2_passive.py [extract|analyse|all]
"""
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\jsche\Desktop\NLP\TEA_Networks")

from study2_reddit import load_merged_study2, hedges_g, bh_adjust, TEXTS, RES
from study2_signatures import head, LABEL_WORDS, GROUPS, SEED
from study2_agency import agent_category

OUT = os.path.join(RES, "signatures")
SLOTS = os.path.join(RES, "slots_with_passive.csv.gz")
SELF = {"i", "me", "myself", "we", "us", "ourselves", "my"}


# ------------------------------------------------------------------ extract
def extract():
    import spacy
    from teanets.svo_extraction import extract_svos

    df = load_merged_study2()
    df = df[df["subreddit"].isin(GROUPS)].reset_index(drop=True)
    texts = []
    for slug in df["slug"]:
        with open(os.path.join(TEXTS, f"{slug}.txt"), encoding="utf-8") as f:
            texts.append(f.read())

    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    rows = []
    t0 = time.time()
    for i, (doc, slug) in enumerate(zip(nlp.pipe(texts, batch_size=32, n_process=4), df["slug"])):
        try:
            svo = extract_svos(doc, semantic_relations=False)
        except Exception:
            continue
        svo = svo[svo["Semantic-Syntactic"] == 0]
        for r in svo.itertuples(index=False):
            role_pair = (r.TEA, r.TEA2)
            if role_pair == ("Agent", "Event"):
                rows.append((slug, head(str(r[0])), "A", head(str(r[2])), int(r.passive_approx), int(r.is_passive)))
            elif role_pair == ("Event", "Target"):
                rows.append((slug, head(str(r[2])), "T", head(str(r[0])), 0, int(r.is_passive)))
        if (i + 1) % 1000 == 0:
            print(f"[extract {i+1}/{len(df)}] {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(rows, columns=["slug", "word", "slot", "verb", "passive_approx", "is_passive"])
    out = out[out["word"].notna() & ~out["word"].isin(LABEL_WORDS)]
    out.to_csv(SLOTS, index=False, compression="gzip")
    print(f"Saved {len(out)} slots from {out.slug.nunique()} posts in {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ analyse
def to_table(slots, posts, corrected):
    s = slots.copy()
    if corrected:
        s.loc[(s.slot == "A") & (s.passive_approx == 1), "slot"] = "T"
    s["category"] = s["word"].map(lambda w: agent_category(w))
    tab = (s.assign(agent=(s.slot == "A").astype(int), target=(s.slot == "T").astype(int))
             .groupby(["slug", "category"])[["agent", "target"]].sum().reset_index())
    tab["subreddit"] = posts.loc[tab["slug"], "subreddit"].to_numpy()
    return tab[["slug", "subreddit", "category", "agent", "target"]]


def passive_profile(slots, posts):
    a = slots[slots.slot == "A"].copy()
    a["is_self"] = a["word"].isin(SELF)
    per = a.groupby("slug").agg(
        subj_slots=("passive_approx", "size"),
        passive_rate=("passive_approx", "mean"),
    )
    self_rows = a[a.is_self].groupby("slug")["passive_approx"].agg(["size", "mean"])
    per["self_passive_rate"] = self_rows["mean"]
    per["self_slots"] = self_rows["size"]
    per["subreddit"] = posts.loc[per.index, "subreddit"].to_numpy()
    rows = []
    for col, min_n in (("passive_rate", "subj_slots"), ("self_passive_rate", "self_slots")):
        d = per[per[min_n] >= 5]
        row = {"measure": col, "n_posts": len(d)}
        for g in GROUPS:
            m = (d.subreddit == g).to_numpy()
            x = d[col].to_numpy()
            row[f"pct_{g}"] = 100 * x[m].mean()
            row[f"g_{g}"] = hedges_g(x[m], x[~m])
        rows.append(row)
    return pd.DataFrame(rows)


def top_passive_verbs(slots, posts, n=10):
    a = slots[(slots.slot == "A") & (slots.passive_approx == 1) & slots.word.isin(SELF)].copy()
    a["subreddit"] = posts.loc[a["slug"], "subreddit"].to_numpy()
    out = {}
    for g in GROUPS:
        vc = a[a.subreddit == g].groupby("verb")["slug"].nunique().sort_values(ascending=False)
        n_posts = (posts.subreddit == g).sum()
        out[g] = ", ".join(f"{v} ({100*c/n_posts:.1f}%)" for v, c in vc.head(n).items())
    return out


def analyse():
    from study2_agentivity import run_tests, descriptives

    df = load_merged_study2()
    df = df[df["subreddit"].isin(GROUPS)].reset_index(drop=True)
    posts = df.set_index("slug")[["subreddit"]]
    slots = pd.read_csv(SLOTS, keep_default_na=False, na_values=[""])  # "none"/"null" are real words
    slots = slots[slots["word"].notna()]
    slots = slots[slots.slug.isin(posts.index)]
    pd.set_option("display.width", 220)

    print("=== Agentless passives in the subject slot ===")
    prof = passive_profile(slots, posts)
    prof.to_csv(os.path.join(OUT, "passive_profile.csv"), index=False)
    print(prof.round(3).to_string(index=False))
    print("\nMost common agentless-passive verbs with I/we as patient (% of posts in group):")
    for g, s in top_passive_verbs(slots, posts).items():
        print(f"  {g:10s} {s}")

    results = {}
    for label, corrected in (("as extracted", False), ("passive-corrected", True)):
        tab = to_table(slots, posts, corrected)
        share = tab.groupby("category")[["agent", "target"]].sum().sum(axis=1)
        cats = [c for c in (share / share.sum()).sort_values(ascending=False).index
                if (share / share.sum())[c] >= 0.01 and c != "not in WordNet"]
        agentivity, _ = descriptives(tab, cats)
        res = run_tests(tab, posts, cats)
        res["scheme"] = label
        results[label] = (tab, res, agentivity)
        print(f"\n=== Agentivity % ({label}) — key categories ===")
        key = ["all words", "self (I/we)", "noun.person", "he/she", "they", "noun.cognition"]
        print(agentivity.loc[[k for k in key if k in agentivity.index]].round(1).to_string())

    both = pd.concat([results[k][1] for k in results], ignore_index=True)
    both.to_csv(os.path.join(OUT, "agentivity_passive_corrected.csv"), index=False)
    cmp_ = (both.pivot_table(index=["category", "group"], columns="scheme", values="OR")
                .reset_index())
    pbh = both[both.scheme == "passive-corrected"].set_index(["category", "group"])["p_bh"]
    cmp_["p_bh_corrected"] = pbh.loc[list(zip(cmp_.category, cmp_.group))].to_numpy()
    focus = cmp_[cmp_.category.isin(["self (I/we)", "noun.person", "he/she", "they",
                                     "noun.cognition", "absolutist pronoun"])]
    print("\n=== Category x disorder agentivity OR: as extracted vs passive-corrected ===")
    print(focus.round(3).to_string(index=False))

    tab_c = results["passive-corrected"][0]
    rng = np.random.default_rng(SEED)
    slugs = df["slug"].to_numpy()
    half = set(slugs[rng.random(len(slugs)) < 0.5])
    cats_c = results["passive-corrected"][1]["category"].unique().tolist()
    r1 = run_tests(tab_c[tab_c.slug.isin(half)], posts, cats_c)
    r2 = run_tests(tab_c[~tab_c.slug.isin(half)], posts, cats_c)
    r = np.corrcoef(np.log(r1.OR), np.log(r2.OR))[0, 1]
    print(f"\nSplit-half (passive-corrected): r of log ORs = {r:.3f}")
    sh = r1[["category", "group", "OR"]].merge(r2[["category", "group", "OR"]],
                                               on=["category", "group"], suffixes=("_h1", "_h2"))
    print(sh[sh.category.isin(["self (I/we)", "noun.person", "they"])].round(3).to_string(index=False))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("extract", "all"):
        extract()
    if stage in ("analyse", "all"):
        analyse()
