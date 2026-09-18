"""
Recurrence Quantification Analysis (RQA) on discourse sequences.

Each text is converted to a sequence of sentence embeddings (spaCy vectors).
A recurrence matrix is built: R(i,j) = 1 if cosine_sim(sent_i, sent_j) > threshold.
RQA metrics are then extracted from the recurrence plot.

Key metrics:
- RR  (Recurrence Rate): how often the discourse revisits a previous semantic state
- DET (Determinism): fraction of recurrent points on diagonal lines (length >= 2)
                      → predictable, repetitive sequential patterns
- LAM (Laminarity): fraction of recurrent points on vertical lines (length >= 2)
                     → discourse gets "stuck" on a topic
- TT  (Trapping Time): mean vertical line length → how long it stays stuck
- L   (Mean Diagonal Length): mean diagonal line length → length of repeated sequences
- Lmax: longest diagonal line → longest sustained repetition
- ENTR: Shannon entropy of diagonal line lengths → complexity of recurrence structure
"""
import os
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import spacy
from scipy import stats

HERE = os.path.dirname(__file__)


def load_nlp():
    print("Loading en_core_web_lg...")
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 50000
    return nlp


def text_to_sentence_embeddings(text, nlp):
    """Convert text to a sequence of normalized sentence vectors."""
    if len(text) > 45000:
        text = text[:45000]
    doc = nlp(text)
    vecs = []
    for sent in doc.sents:
        if sent.vector_norm > 0 and len(sent.text.split()) >= 3:
            vecs.append(sent.vector / sent.vector_norm)
    return np.array(vecs) if vecs else None


def build_recurrence_matrix(vecs, threshold):
    """Build binary recurrence matrix from cosine similarities."""
    n = len(vecs)
    sim = vecs @ vecs.T  # cosine similarity (vectors are normalized)
    R = (sim >= threshold).astype(np.int8)
    np.fill_diagonal(R, 0)  # exclude Line of Identity
    return R


def compute_rqa(R):
    """Extract RQA metrics from a binary recurrence matrix."""
    n = R.shape[0]
    total_possible = n * (n - 1)  # exclude diagonal
    total_recurrent = R.sum()

    if total_possible == 0 or total_recurrent == 0:
        return None

    RR = total_recurrent / total_possible

    # Diagonal lines (parallel to main diagonal, length >= 2)
    diag_lengths = []
    for k in range(1, n):  # diagonals above main
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

    for k in range(-n + 1, 0):  # diagonals below main
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

    # Entropy of diagonal line length distribution
    if diag_lengths:
        counts = {}
        for dl in diag_lengths:
            counts[dl] = counts.get(dl, 0) + 1
        total_lines = len(diag_lengths)
        ENTR = -sum((c / total_lines) * np.log2(c / total_lines)
                     for c in counts.values())
    else:
        ENTR = 0

    # Vertical lines (length >= 2) → Laminarity and Trapping Time
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
    """Find cosine similarity threshold that gives ~target_rr recurrence rate.
    Uses a sample of texts to calibrate."""
    all_sims = []
    for vecs in all_vecs[:20]:
        if vecs is not None and len(vecs) > 5:
            sim = vecs @ vecs.T
            np.fill_diagonal(sim, 0)
            upper = sim[np.triu_indices_from(sim, k=1)]
            all_sims.extend(upper.tolist())

    if not all_sims:
        return 0.85

    all_sims = np.array(all_sims)
    threshold = np.percentile(all_sims, (1 - target_rr) * 100)
    return threshold


def process_corpus(name, texts_dir, metadata_path, nlp, threshold, max_n=200):
    import random
    random.seed(42)

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
            print(f"  [{name}] [{i+1}/{len(meta)}] {slug} "
                  f"(sents={rqa['n_sentences']}, RR={rqa['RR']:.3f}, DET={rqa['DET']:.3f}, LAM={rqa['LAM']:.3f})")

    return results


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def main():
    nlp = load_nlp()

    # First pass: get sentence embeddings to calibrate threshold
    print("Calibrating threshold...")
    sample_vecs = []
    ocd_meta = []
    with open(os.path.join(HERE, "metadata.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ocd_meta.append(row)
    for row in ocd_meta[:25]:
        txt_path = os.path.join(HERE, "texts", f"{row['slug']}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8") as f:
                text = f.read()
            vecs = text_to_sentence_embeddings(text, nlp)
            if vecs is not None:
                sample_vecs.append(vecs)

    ctrl_meta = []
    with open(os.path.join(HERE, "control_gutenberg", "metadata.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("word_count", 0)) >= 200:
                ctrl_meta.append(row)
    for row in ctrl_meta[:25]:
        txt_path = os.path.join(HERE, "control_gutenberg", "texts", f"{row['slug']}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8") as f:
                text = f.read()
            if len(text) <= 45000:
                vecs = text_to_sentence_embeddings(text, nlp)
                if vecs is not None:
                    sample_vecs.append(vecs)

    threshold = find_threshold(sample_vecs, target_rr=0.10)
    print(f"  Threshold (target RR~10%): {threshold:.4f}\n")

    print("Processing OCD corpus...")
    ocd = process_corpus(
        "OCD",
        os.path.join(HERE, "texts"),
        os.path.join(HERE, "metadata.csv"),
        nlp, threshold
    )

    print(f"\nProcessing Control corpus...")
    ctrl = process_corpus(
        "Control",
        os.path.join(HERE, "control_gutenberg", "texts"),
        os.path.join(HERE, "control_gutenberg", "metadata.csv"),
        nlp, threshold
    )

    # Compare
    metrics = [
        ("RR", "Recurrence Rate"),
        ("DET", "Determinism"),
        ("LAM", "Laminarity (getting stuck)"),
        ("TT", "Trapping Time"),
        ("L", "Mean Diagonal Length"),
        ("Lmax", "Max Diagonal Length"),
        ("ENTR", "Recurrence Entropy"),
    ]

    print("\n" + "=" * 105)
    print(f"  {'Metric':38s} {'OCD (M+/-SD)':18s} {'Ctrl (M+/-SD)':18s} {'U':>8s} {'p':>10s} {'d':>7s} {'Eff':>10s}")
    print("=" * 105)

    all_results = []
    for col, label in metrics:
        ocd_vals = np.array([r[col] for r in ocd if r.get(col) is not None])
        ctrl_vals = np.array([r[col] for r in ctrl if r.get(col) is not None])

        if len(ocd_vals) < 5 or len(ctrl_vals) < 5:
            print(f"  {label:38s} insufficient data")
            continue

        ocd_m, ocd_s = np.mean(ocd_vals), np.std(ocd_vals)
        ctrl_m, ctrl_s = np.mean(ctrl_vals), np.std(ctrl_vals)
        u, p = stats.mannwhitneyu(ocd_vals, ctrl_vals, alternative="two-sided")
        d = cohens_d(ocd_vals, ctrl_vals)
        d_abs = abs(d)
        eff = "negligible" if d_abs < 0.2 else "small" if d_abs < 0.5 else "medium" if d_abs < 0.8 else "large"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"  {label:38s} {ocd_m:.4f}+/-{ocd_s:.4f}  {ctrl_m:.4f}+/-{ctrl_s:.4f}  "
              f"{u:8.0f} {p:10.6f}{sig:3s} {d:+7.3f} {eff:>10s}")

        all_results.append({
            "metric": label, "ocd_mean": round(ocd_m, 4), "ocd_std": round(ocd_s, 4),
            "ctrl_mean": round(ctrl_m, 4), "ctrl_std": round(ctrl_s, 4),
            "U": u, "p_value": round(p, 6), "cohens_d": round(d, 3), "effect": eff,
        })

    print("=" * 105)
    print(f"\n  OCD: n={len(ocd)}, Control: n={len(ctrl)}")
    print(f"  Cosine threshold: {threshold:.4f}")

    # Save per-text RQA
    for name, data in [("ocd", ocd), ("control", ctrl)]:
        path = os.path.join(HERE, "results", f"rqa_{name}.csv")
        if data:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)

    # Save comparison
    if all_results:
        path = os.path.join(HERE, "results", "rqa_comparison.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)

    print("  Saved to results/rqa_*.csv")


if __name__ == "__main__":
    main()
