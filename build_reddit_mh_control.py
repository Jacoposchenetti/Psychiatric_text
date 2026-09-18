"""
Build mental health control corpus from Reddit posts (depression + PTSD).
Source: solomonk/reddit_mental_health_posts on HuggingFace.
Filters for 300+ word posts to match OCD Stories length.
"""
import os
import csv
import random
import re
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
import pandas as pd
import numpy as np

random.seed(42)

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "control_mental_health")
TEXTS_DIR = os.path.join(OUT_DIR, "texts")
os.makedirs(TEXTS_DIR, exist_ok=True)


def clean_reddit_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#x200B;", "", text)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[>\-\*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"edit:.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"tl;?dr.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def main():
    print("Loading dataset...")
    ds = load_dataset("solomonk/reddit_mental_health_posts", split="train")
    df = pd.DataFrame(ds)

    print(f"Total posts: {len(df)}")
    print(f"\nSubreddit distribution:")
    print(df["subreddit"].value_counts())

    # Filter: depression + ptsd, 300+ words
    mh_control = df[df["subreddit"].isin(["depression", "ptsd"])].copy()
    mh_control["body"] = mh_control["body"].fillna("")
    mh_control["clean_text"] = mh_control["body"].apply(clean_reddit_text)
    mh_control["wc"] = mh_control["clean_text"].str.split().str.len()

    long_posts = mh_control[mh_control["wc"] >= 300].copy()
    print(f"\nDepression+PTSD posts with 300+ words: {len(long_posts)}")
    print(f"  depression: {(long_posts['subreddit'] == 'depression').sum()}")
    print(f"  ptsd: {(long_posts['subreddit'] == 'ptsd').sum()}")

    # Sample 200 posts (more than needed, we'll match 152 later)
    n_sample = min(200, len(long_posts))
    sample = long_posts.sample(n_sample, random_state=42)

    print(f"\nSampled: {len(sample)}")
    print(f"  Word count: mean={sample['wc'].mean():.0f}, median={sample['wc'].median():.0f}, "
          f"min={sample['wc'].min()}, max={sample['wc'].max()}")

    # Save
    metadata = []
    for _, row in sample.iterrows():
        slug = f"{row['subreddit']}_{row['id']}"
        text = row["clean_text"]

        # Prepend title
        title = str(row.get("title", "")).strip()
        if title and title.lower() != "nan":
            text = title + "\n\n" + text

        txt_path = os.path.join(TEXTS_DIR, f"{slug}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        metadata.append({
            "slug": slug,
            "title": title,
            "subreddit": row["subreddit"],
            "author": row.get("author", ""),
            "date": str(row.get("created_utc", "")),
            "score": row.get("score", 0),
            "word_count": len(text.split()),
            "url": row.get("url", ""),
        })

    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metadata[0].keys()))
        w.writeheader()
        w.writerows(metadata)

    print(f"\nSaved {len(metadata)} posts to {OUT_DIR}")

    wcs = np.array([m["word_count"] for m in metadata])
    print(f"\nFinal word count stats:")
    print(f"  Mean: {wcs.mean():.0f}")
    print(f"  Median: {np.median(wcs):.0f}")
    print(f"  Min: {wcs.min()}, Max: {wcs.max()}")
    print(f"  Total: {wcs.sum():,}")

    # Subreddit split
    from collections import Counter
    subs = Counter(m["subreddit"] for m in metadata)
    print(f"\nSubreddit split:")
    for s, c in subs.most_common():
        print(f"  r/{s}: {c}")


if __name__ == "__main__":
    main()
