"""
RQA (Recurrence Quantification Analysis) on all three groups:
OCD Stories, Cancer Patient Stories, Depression/PTSD Reddit posts.
Calibrates a single threshold across all groups, then compares.
"""
import os
import csv
import warnings
import random
warnings.filterwarnings("ignore")

import numpy as np
import spacy
from scipy import stats

random.seed(42)
HERE = os.path.dirname(__file__)


def load_nlp():
    print("Loading en_core_web_lg...")
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    return nlp


def text_to_sentence_embeddings(text, nlp):
    if len(text) > 45000:
        text = text[:45000]
    doc = nlp(text)
    vecs = []
    for sent in doc.sents:
        if sent.vector_norm > 0 and len(sent.text.split()) >= 3:
            vecs.append(sent.vector / sent.vector_norm)
    return np.array(vecs) if vecs else None


def build_recurrence_matrix(vecs, threshold):
    sim = vecs @ vecs.T
    R = (sim >= threshold).astype(np.int8)
    np.fill_diagonal(R, 0)
    return R


def compute_rqa(R):
    n = R.shape[0]
    total_possible = n * (n - 1)
    total_recurrent = R.sum()
    if total_possible == 0 or total_recurrent == 0:
        return None

    RR = total_recurrent / total_possible

    diag_lengths = []
    for k in list(range(1, n)) + list(range(-n + 1, 0)):
        diag = np.diag(R, k)
        length = 0
        for val in diag:
            if val:
                length += 1
            else:
                if length >= 2:
                    diag_lengths.append(length)
                length = 0
        if length >= 2:
            diag_lengths.append(length)

    points_on_diag = sum(diag_lengths) if diag_lengths else 0
    DET = points_on_diag / total_recurrent if total_recurrent > 0 else 0
    L = np.mean(diag_lengths) if diag_lengths else 0
    Lmax = max(diag_lengths) if diag_lengths else 0

    if diag_lengths:
        counts = {}
        for dl in diag_lengths:
            counts[dl] = counts.get(dl, 0) + 1
        total_lines = len(diag_lengths)
        ENTR = -sum((c / total_lines) * np.log2(c / total_lines) for c in counts.values())
    else:
        ENTR = 0

    vert_lengths = []
    for col in range(n):
        length = 0
        for row in range(n):
            if R[row, col]:
                length += 1
            else:
                if length >= 2:
                    vert_lengths.append(length)
                length = 0
        if length >= 2:
            vert_lengths.append(length)

    points_on_vert = sum(vert_lengths) if vert_lengths else 0
    LAM = points_on_vert / total_recurrent if total_recurrent > 0 else 0
    TT = np.mean(vert_lengths) if vert_lengths else 0

    return {
        "n_sentences": n,
        "RR": RR,
        "DET": DET,
        "LAM": LAM,
        "TT": TT,
        "L": L,
        "Lmax": Lmax,
        "ENTR": ENTR,
    }


def find_threshold(all_vecs, target_rr=0.10):
    all_sims = []
    for vecs in all_vecs[:30]:
        if vecs is not None and len(vecs) > 5:
            sim = vecs @ vecs.T
            np.fill_diagonal(sim, 0)
            upper = sim[np.triu_indices_from(sim, k=1)]
            all_sims.extend(upper.tolist())
    if not all_sims:
        return 0.85
    return np.percentile(np.array(all_sims), (1 - target_rr) * 100)


def process_corpus(name, texts_dir, metadata_path, nlp, threshold, max_n=200):
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

        vecs = text_to_sentence_embeddings(text, nlp)
        if vecs is None or len(vecs) < 10:
            continue

        R = build_recurrence_matrix(vecs, threshold)
        rqa = compute_rqa(R)
        if rqa is None:
            continue

        rqa["slug"] = slug
        rqa["n_words"] = len(text.split())
        results.append(rqa)

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{name}] [{i+1}/{len(meta)}] sents={rqa['n_sentences']}, "
                  f"RR={rqa['RR']:.3f}, DET={rqa['DET']:.3f}")

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
    b_avail = list(range(len(group_b)))
    b_words = np.array([r["n_words"] for r in group_b])
    matched_a, matched_b = [], []

    indices_a = list(range(len(group_a)))
    random.shuffle(indices_a)

    used_b = set()
    for ia in indices_a:
        wc_a = group_a[ia]["n_words"]
        best_ib = None
        best_diff = float("inf")
        for ib in b_avail:
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


def compare_rqa(name_a, data_a, name_b, data_b, metrics):
    matched_a, matched_b = match_by_word_count(data_a, data_b)

    if len(matched_a) < 15:
        print(f"  Too few matched pairs ({len(matched_a)}) for {name_a} vs {name_b}")
        return []

    wc_a = np.array([r["n_words"] for r in matched_a])
    wc_b = np.array([r["n_words"] for r in matched_b])
    d_wc = cohens_d(wc_a, wc_b)

    print(f"\n  {name_a} vs {name_b}: {len(matched_a)} matched pairs")
    print(f"  {name_a} words: mean={wc_a.mean():.0f}, {name_b} words: mean={wc_b.mean():.0f}, d_wc={d_wc:+.3f}")

    print(f"\n  {'Metric':35s} {name_a:>10s} {name_b:>10s} {'d':>8s} {'p':>10s} {'Eff':>8s}")
    print("  " + "-" * 85)

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

        print(f"  {label:35s} {np.mean(vals_a):10.4f} {np.mean(vals_b):10.4f} "
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

    # Calibrate threshold on a sample from all three groups
    print("Calibrating threshold on combined sample...")
    sample_vecs = []

    for texts_dir, meta_path in [
        (os.path.join(HERE, "texts"), os.path.join(HERE, "metadata.csv")),
        (os.path.join(HERE, "control_patient_stories", "texts"),
         os.path.join(HERE, "control_patient_stories", "metadata.csv")),
        (os.path.join(HERE, "control_mental_health", "texts"),
         os.path.join(HERE, "control_mental_health", "metadata.csv")),
    ]:
        meta = []
        with open(meta_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta.append(row)
        for row in meta[:10]:
            txt_path = os.path.join(texts_dir, f"{row['slug']}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, encoding="utf-8") as f:
                    text = f.read()
                if len(text.split()) >= 200:
                    vecs = text_to_sentence_embeddings(text, nlp)
                    if vecs is not None:
                        sample_vecs.append(vecs)

    threshold = find_threshold(sample_vecs, target_rr=0.10)
    print(f"  Threshold (target RR~10%): {threshold:.4f}\n")

    # Process each group
    print("Processing OCD Stories...")
    ocd = process_corpus("OCD", os.path.join(HERE, "texts"),
                         os.path.join(HERE, "metadata.csv"), nlp, threshold)

    print("\nProcessing Cancer Patient Stories...")
    cancer = process_corpus("Cancer", os.path.join(HERE, "control_patient_stories", "texts"),
                            os.path.join(HERE, "control_patient_stories", "metadata.csv"),
                            nlp, threshold, max_n=152)

    print("\nProcessing Depression/PTSD Reddit...")
    mh = process_corpus("Dep/PTSD", os.path.join(HERE, "control_mental_health", "texts"),
                        os.path.join(HERE, "control_mental_health", "metadata.csv"),
                        nlp, threshold, max_n=200)

    print(f"\n  Group sizes: OCD={len(ocd)}, Cancer={len(cancer)}, Dep/PTSD={len(mh)}")

    rqa_metrics = [
        ("RR", "Recurrence Rate"),
        ("DET", "Determinism"),
        ("LAM", "Laminarity (stuck on topic)"),
        ("TT", "Trapping Time"),
        ("L", "Mean Diagonal Length"),
        ("Lmax", "Max Diagonal (longest repeat)"),
        ("ENTR", "Recurrence Entropy"),
    ]

    all_results = []

    print("\n" + "=" * 100)
    print("  RQA THREE-GROUP COMPARISON (length-matched)")
    print("=" * 100)

    all_results.extend(compare_rqa("OCD", ocd, "Cancer", cancer, rqa_metrics))
    print()
    all_results.extend(compare_rqa("OCD", ocd, "Dep/PTSD", mh, rqa_metrics))

    # Summary
    print("\n" + "=" * 100)
    print("  SIGNIFICANT RQA FINDINGS")
    print("=" * 100)

    sig = [r for r in all_results if r["significant"]]
    if sig:
        for r in sorted(sig, key=lambda x: -abs(x["cohens_d"])):
            dir_ = "HIGHER" if r["cohens_d"] > 0 else "LOWER"
            comp = r["comparison"]
            print(f"  {comp:30s}: {r['metric']:35s} d={r['cohens_d']:+.3f} ({r['effect']}, p={r['p_value']:.6f})")
    else:
        print("  No significant RQA differences found.")

    # Save per-group RQA
    for name, data in [("ocd", ocd), ("cancer", cancer), ("mh", mh)]:
        if data:
            path = os.path.join(HERE, "results", f"rqa_{name}.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)

    if all_results:
        path = os.path.join(HERE, "results", "rqa_three_group_comparison.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\n  Saved to results/rqa_*.csv")


if __name__ == "__main__":
    main()
