"""
Clean scraped patient stories: remove boilerplate, fix encoding, filter by word count.
"""
import os
import csv
import re

HERE = os.path.dirname(__file__)
CORPUS_DIR = os.path.join(HERE, "control_patient_stories")
TEXTS_DIR = os.path.join(CORPUS_DIR, "texts")

BOILERPLATE = [
    r"Interviewed by:.*?(?:Edited by:.*?\n|\n)",
    r"Edited by:.*?\n",
    r"Watch .+?video .+?(?:below|above)\.?\s*",
    r"(?:You.ll find out more about .+?story\.?\s*)",
    r".+?Diagnosis Facts\s*",
    r"Sponsored by.*?\n",
    r"This interview has been edited.*?\n",
    r"More .+? Stories\s*",
    r"Share Your .+? Story\s*",
    r"(?:Get|Join|Sign up for) (?:our|the) newsletter.*?\n",
    r"Updated:?\s*\w+ \d{1,2},?\s*\d{4}\s*",
    r"(?:Written|Published|Posted) (?:by|on).*?\n",
]


def clean_text(text):
    # Fix common unicode issues
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("…", "...")
    text = text.replace(" ", " ")
    # Mojibake patterns (UTF-8 read as Windows-1252)
    text = text.replace("â", "'")
    text = text.replace("â", '"')
    text = text.replace("â", '"')
    text = text.replace("â", "-")
    text = text.replace("â", "-")
    text = text.replace("â¦", "...")

    for pattern in BOILERPLATE:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if len(line) < 15:
            continue
        cleaned.append(line)

    text = "\n\n".join(p for p in cleaned if p)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    meta_path = os.path.join(CORPUS_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        print("No metadata.csv found")
        return

    meta = []
    with open(meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta.append(row)

    print(f"Total stories in metadata: {len(meta)}")

    cleaned_meta = []
    skipped = 0
    for row in meta:
        slug = row["slug"]
        txt_path = os.path.join(TEXTS_DIR, f"{slug}.txt")
        if not os.path.exists(txt_path):
            skipped += 1
            continue

        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        text = clean_text(text)
        wc = len(text.split())

        if wc < 300:
            skipped += 1
            continue

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        row["word_count"] = str(wc)
        cleaned_meta.append(row)

    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "title", "cancer_type", "url", "word_count"])
        w.writeheader()
        w.writerows(cleaned_meta)

    print(f"After cleaning: {len(cleaned_meta)} stories (skipped {skipped})")

    wcs = [int(r["word_count"]) for r in cleaned_meta]
    import numpy as np
    wcs = np.array(wcs)
    print(f"\nWord count stats:")
    print(f"  Mean: {wcs.mean():.0f}")
    print(f"  Median: {np.median(wcs):.0f}")
    print(f"  Min: {wcs.min()}, Max: {wcs.max()}")
    print(f"  Total: {wcs.sum():,}")

    from collections import Counter
    types = Counter(r["cancer_type"] for r in cleaned_meta)
    print(f"\nCancer type distribution:")
    for t, c in types.most_common():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
