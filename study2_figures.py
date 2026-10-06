"""Figures for the agency paper (Study 2, Reddit)."""
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\jsche\Desktop\NLP\TEA_Networks")

from study2_reddit import load_merged_study2, TEXTS, RES
from study2_signatures import head, LABEL_WORDS, GROUPS, SEED
from study2_passive import to_table, SLOTS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "paper_agency", "figures")
os.makedirs(FIG, exist_ok=True)
SIG = os.path.join(RES, "signatures")

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = "#2a78d6"
BLUE, MID, RED = "#2a78d6", "#f0efec", "#e34948"
NAMES = {"OCD": "OCD", "depression": "Depression", "ptsd": "PTSD", "ADHD": "ADHD"}
CAT_LABEL = {
    "self (I/we)": "Self (I, we)", "noun.person": "Persons (nouns)", "he/she": "he / she",
    "they": "they", "noun.cognition": "Cognition nouns", "absolutist pronoun": "Absolutist pronouns",
    "noun.time": "Time nouns", "noun.artifact": "Artifact nouns", "noun.body": "Body nouns",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def hgrid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------ Fig 1: TEA example
def fig_example():
    import spacy
    from teanets.svo_extraction import extract_svos
    from teanets.teaplot import plot_svo_graph
    nlp = spacy.load("en_core_web_lg")
    text = ("He yelled at me again. I was assaulted last year. "
            "My thoughts tell me that I am dangerous. I check the lock every night. "
            "Nothing helps anymore. The medication helps my focus.")
    df = extract_svos(nlp(text), semantic_relations=False)
    plot_svo_graph(df, custom_font=13, filename=os.path.join(FIG, "fig1_tea_example.png"),
                   mark_passive_approx=True, seed=SEED, show=False)
    df.to_csv(os.path.join(FIG, "fig1_tea_example_triplets.csv"), index=False)


# --------------------------------------- Fig 2: data-driven TEA networks
LIGHT_VERBS = {"be", "do", "have", "get"}
PARTICLES = {"back", "up", "out", "off", "away", "down", "over", "around"}
GENERIC_TARGETS = {"it", "what", "that", "this", "thing", "something", "anything", "everything",
                   "time", "one", "way", "lot", "there"}
DISPLAY_FIX = {"rap": "rape"}  # spaCy lemmatises "raped" as "rap"


def count_triples(group, subject):
    import spacy
    from nltk.stem import WordNetLemmatizer
    from teanets.svo_extraction import extract_svos
    lem = WordNetLemmatizer()
    df = load_merged_study2()
    slugs = df[df.subreddit == group]["slug"].tolist()
    texts = [open(os.path.join(TEXTS, f"{s}.txt"), encoding="utf-8").read() for s in slugs]
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000

    def lemma(phrase, pos):
        h = head(str(phrase))
        return lem.lemmatize(h, pos) if h else None

    triples = Counter()
    for doc in nlp.pipe(texts, batch_size=32, n_process=4):
        svo = extract_svos(doc, semantic_relations=False)
        svo = svo[svo["Semantic-Syntactic"] == 0]
        seen = set()
        for _, rows in svo.groupby("svo_id"):
            ag = rows[(rows.TEA == "Agent") & (rows.TEA2 == "Event")]
            if ag.empty or lemma(ag.iloc[0]["Node 1"], "n") != subject:
                continue
            if ag["passive_approx"].astype(int).max() == 1:
                continue
            ev = lemma(ag.iloc[0]["Node 2"], "v")
            tg = rows[(rows.TEA == "Event") & (rows.TEA2 == "Target")]
            for t in [lemma(t, "n") for t in tg["Node 2"]] or [""]:
                if t not in LABEL_WORDS:
                    seen.add((ev, t or ""))
        triples.update(seen)
    out = pd.DataFrame([(e, t, c) for (e, t), c in triples.items()], columns=["event", "target", "n_posts"])
    out = out.sort_values("n_posts", ascending=False)
    out.to_csv(os.path.join(FIG, f"fig2_{group}_{subject}_counts.csv"), index=False)
    print(f"{group}/{subject}: {len(slugs)} posts, {len(out)} distinct (event, target) pairs")


def plot_triples(group, subject, top=12, min_posts=5):
    from teanets.teaplot import plot_svo_graph
    c = pd.read_csv(os.path.join(FIG, f"fig2_{group}_{subject}_counts.csv"), keep_default_na=False)
    from nltk.corpus import wordnet as wn
    # the event head is the last token of the verb phrase, which is sometimes an adverb ("again")
    is_verb = c.event.map(lambda e: bool(wn.synsets(e, pos=wn.VERB)) and e not in PARTICLES)
    c = c[(c.target != "") & is_verb & ~c.event.isin(LIGHT_VERBS) & ~c.target.isin(GENERIC_TARGETS)
          & (c.n_posts >= min_posts)].head(top)
    rows = []
    for i, r in enumerate(c.itertuples()):
        ev = DISPLAY_FIX.get(r.event, r.event)
        hg = str([[subject], [ev], [r.target]])
        rows.append({"Node 1": subject, "TEA": "Agent", "Node 2": ev, "TEA2": "Event",
                     "Hypergraph": hg, "Semantic-Syntactic": 0, "svo_id": i, "passive_approx": 0})
        rows.append({"Node 1": ev, "TEA": "Event", "Node 2": r.target, "TEA2": "Target",
                     "Hypergraph": hg, "Semantic-Syntactic": 0, "svo_id": i, "passive_approx": 0})
    pd.DataFrame(rows).to_csv(os.path.join(FIG, f"fig2_{group}_{subject}_plotted.csv"), index=False)
    c.to_csv(os.path.join(FIG, f"fig2_{group}_{subject}_plotted_counts.csv"), index=False)
    plot_svo_graph(pd.DataFrame(rows), subject_filter=subject, custom_font=14,
                   filename=os.path.join(FIG, f"fig2_tea_{group}_{subject}.png"), seed=SEED, show=False)
    print(c.to_string(index=False))


def fig_real_networks_all():
    for group, subject in (("ptsd", "he"), ("OCD", "thought")):
        if not os.path.exists(os.path.join(FIG, f"fig2_{group}_{subject}_counts.csv")):
            count_triples(group, subject)
        plot_triples(group, subject)


# --------------------------------- Fig 3: mention vs agentivity (bootstrap)
def fig_mention_agentivity(n_boot=1000):
    df = load_merged_study2()
    posts = df.set_index("slug")[["subreddit"]]
    slots = pd.read_csv(SLOTS, keep_default_na=False, na_values=[""])
    slots = slots[slots.word.notna() & slots.slug.isin(posts.index)]
    tab = to_table(slots, posts, corrected=True)
    tot = tab.groupby("slug")[["agent", "target"]].sum().sum(axis=1)
    cats = ["self (I/we)", "noun.person", "they"]
    rng = np.random.default_rng(SEED)
    rows = []
    for c in cats:
        sub = tab[tab.category == c].set_index("slug")[["agent", "target"]]
        for g in GROUPS:
            ids = posts.index[posts.subreddit == g]
            a = sub["agent"].reindex(ids, fill_value=0).to_numpy(float)
            t = sub["target"].reindex(ids, fill_value=0).to_numpy(float)
            n = tot.reindex(ids, fill_value=0).to_numpy(float)
            boots_m, boots_a = [], []
            for _ in range(n_boot):
                k = rng.integers(0, len(ids), len(ids))
                boots_m.append((a[k] + t[k]).sum() / n[k].sum())
                boots_a.append(a[k].sum() / (a[k] + t[k]).sum())
            rows.append({"category": c, "group": g,
                         "mention": 100 * (a + t).sum() / n.sum(),
                         "mention_lo": 100 * np.percentile(boots_m, 2.5), "mention_hi": 100 * np.percentile(boots_m, 97.5),
                         "agent": 100 * a.sum() / (a + t).sum(),
                         "agent_lo": 100 * np.percentile(boots_a, 2.5), "agent_hi": 100 * np.percentile(boots_a, 97.5)})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(SIG, "mention_vs_agentivity.csv"), index=False)

    fig, axes = plt.subplots(len(cats), 2, figsize=(6.6, 5.2), sharey="row")
    for i, c in enumerate(cats):
        s = d[d.category == c].set_index("group").loc[GROUPS[::-1]]
        y = np.arange(len(GROUPS))
        for j, (m, lo, hi, lab) in enumerate((("mention", "mention_lo", "mention_hi", "Mention rate (% of slots)"),
                                              ("agent", "agent_lo", "agent_hi", "Agentivity (% as Agent)"))):
            ax = axes[i, j]
            ax.hlines(y, s[lo], s[hi], color=SERIES, linewidth=2)
            ax.scatter(s[m], y, s=36, color=SERIES, edgecolor="white", linewidth=1.5, zorder=3)
            for yy, v in zip(y, s[m]):
                ax.annotate(f"{v:.1f}", (v, yy), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK2)
            ax.set_yticks(y, [NAMES[g] for g in GROUPS[::-1]])
            ax.set_ylim(-0.6, len(GROUPS) - 0.3)
            lo_x, hi_x = s[lo].min(), s[hi].max()
            pad = (hi_x - lo_x) * 0.25 + 0.2
            ax.set_xlim(lo_x - pad, hi_x + pad)
            hgrid(ax)
            if i == 0:
                ax.set_title(lab, fontsize=9, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(CAT_LABEL[c], fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_mention_vs_agentivity.png"))
    plt.close(fig)


# ----------------------------------------------- Fig 4: OR heatmap
def fig_heatmap():
    r = pd.read_csv(os.path.join(SIG, "agentivity_final_author_clustered.csv"))
    order = ["self (I/we)", "he/she", "they", "noun.person", "absolutist pronoun",
             "noun.cognition", "noun.time", "noun.body", "noun.artifact"]
    lor = r.pivot(index="category", columns="group", values="OR").loc[order, GROUPS].apply(np.log2)
    p = r.pivot(index="category", columns="group", values="p_bh").loc[order, GROUPS]
    orv = r.pivot(index="category", columns="group", values="OR").loc[order, GROUPS]
    lim = np.abs(lor.to_numpy()).max()
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, MID, RED])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(lor.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(order)):
        for j in range(len(GROUPS)):
            star = "*" if p.iloc[i, j] < .05 else ""
            ax.text(j, i, f"{orv.iloc[i, j]:.2f}{star}", ha="center", va="center", fontsize=8.5,
                    color=INK if abs(lor.iloc[i, j]) < 0.75 * lim else "white")
    ax.set_xticks(range(len(GROUPS)), [NAMES[g] for g in GROUPS])
    ax.set_yticks(range(len(order)), [CAT_LABEL[c] for c in order])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(GROUPS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(order)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    ticks = [0.5, 0.67, 1, 1.5, 2]
    cb.set_ticks(np.log2(ticks), labels=[str(t) for t in ticks])
    cb.set_label("Agentivity odds ratio (log scale)", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_agentivity_heatmap.png"))
    plt.close(fig)


# ----------------------------------------------- Fig 5: agency map
def fig_agency_map():
    r = pd.read_csv(os.path.join(SIG, "agentivity_final_author_clustered.csv")).set_index(["category", "group"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.axhline(1, color=AXIS, linewidth=0.8)
    ax.axvline(1, color=AXIS, linewidth=0.8)
    offsets = {"OCD": (8, 6), "depression": (8, -12), "ptsd": (9, -13), "ADHD": (8, -12)}
    for g in GROUPS:
        x, y = r.loc[("self (I/we)", g)], r.loc[("noun.person", g)]
        ax.errorbar(x.OR, y.OR, xerr=[[x.OR - x.ci_low], [x.ci_high - x.OR]],
                    yerr=[[y.OR - y.ci_low], [y.ci_high - y.OR]], fmt="o", color=INK,
                    ms=6, mec="white", mew=1.5, elinewidth=1.2, capsize=0)
        ax.annotate(NAMES[g], (x.OR, y.OR), xytext=offsets[g], textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    t = [0.6, 0.8, 1, 1.25, 1.5]
    ax.set_xticks(t, [str(v) for v in t])
    ax.set_yticks(t, [str(v) for v in t])
    ax.minorticks_off()
    ax.set_xlim(0.55, 1.6)
    ax.set_ylim(0.55, 1.75)
    ax.set_xlabel("Self as Agent (odds ratio)")
    ax.set_ylabel("Other persons as Agent (odds ratio)")
    kw = dict(fontsize=7.5, color=MUTED)
    ax.text(0.57, 1.72, "others act,\nself acted upon", va="top", **kw)
    ax.text(1.57, 0.57, "self acts,\nothers in background", ha="right", va="bottom", **kw)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_agency_map.png"))
    plt.close(fig)


# ----------------------------------- Fig 6: content signatures (log-odds)
def fig_content(k=10):
    lo = pd.read_csv(os.path.join(SIG, "content_log_odds.csv"), keep_default_na=False, na_values=[""])
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.4))
    for row, role in enumerate(["Agent", "Event"]):
        for col, g in enumerate(GROUPS):
            ax = axes[row, col]
            top = lo[(lo.group == g) & (lo.role == role)].nlargest(k, "z").iloc[::-1]
            ax.barh(range(len(top)), top["z"], color=SERIES, height=0.7)
            ax.set_yticks(range(len(top)), [DISPLAY_FIX.get(t, t) for t in top["term"]], fontsize=8)
            ax.tick_params(axis="y", length=0)
            ax.set_xlim(0, lo[(lo.role == role)].groupby("group")["z"].max().max() * 1.05)
            hgrid(ax)
            if row == 0:
                ax.set_title(NAMES[g], fontsize=9.5, color=INK)
            if col == 0:
                ax.set_ylabel(f"{role} slot", fontsize=9, color=INK)
            if row == 1:
                ax.set_xlabel("log-odds z", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_content_signatures.png"))
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["example", "real", "mention", "heatmap", "map", "content"]
    funcs = {"example": fig_example, "real": fig_real_networks_all, "mention": fig_mention_agentivity,
             "heatmap": fig_heatmap, "map": fig_agency_map, "content": fig_content}
    for w in which:
        print(f"-- {w}", flush=True)
        funcs[w]()
