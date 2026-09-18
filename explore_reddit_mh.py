"""Explore the Reddit mental health posts dataset from HuggingFace."""
from datasets import load_dataset
import pandas as pd

ds = load_dataset("solomonk/reddit_mental_health_posts", split="train")
df = pd.DataFrame(ds)
print(f"Total posts: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Check what columns exist
print(f"\nFirst row sample:")
print(df.iloc[0].to_dict())

# Distribution by subreddit/label
for col in ["subreddit", "label", "category"]:
    if col in df.columns:
        print(f"\n{col} distribution:")
        print(df[col].value_counts())

# Word count
text_col = None
for c in ["text", "selftext", "post", "body"]:
    if c in df.columns:
        text_col = c
        break
if text_col:
    df["wc"] = df[text_col].fillna("").str.split().str.len()
    print(f"\nWord count stats (column: {text_col}):")
    print(df["wc"].describe())
    print(f"\nPosts with 300+ words: {(df['wc'] >= 300).sum()}")
    print(f"Posts with 500+ words: {(df['wc'] >= 500).sum()}")
    print(f"Posts with 1000+ words: {(df['wc'] >= 1000).sum()}")
