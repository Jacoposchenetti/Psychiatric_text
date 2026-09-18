"""Retry scraping stories that failed in the first pass."""
import requests
from bs4 import BeautifulSoup
import csv
import os
import time

OUT_DIR = os.path.dirname(__file__)
TEXTS_DIR = os.path.join(OUT_DIR, "texts")
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)"}

# All 153 story slugs from the listing pages
ALL_SLUGS = []

# Read metadata to find already-scraped slugs
scraped = set()
csv_path = os.path.join(OUT_DIR, "metadata.csv")
existing_rows = []
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        scraped.add(row["slug"])
        existing_rows.append(row)

# The failed stories (92-153) based on the error log
FAILED_URLS = [
    "https://theocdstories.com/stories/you-deserve-to-be-happy/",
    "https://theocdstories.com/stories/dear-ocd/",
    "https://theocdstories.com/stories/a-mothers-journey-helping-her-son-recover-from-ocd/",
    "https://theocdstories.com/stories/the-daily-struggle-have-hope/",
    "https://theocdstories.com/stories/the-ocd-rabbit-hole/",
    "https://theocdstories.com/stories/the-ocd-nightmare/",
    "https://theocdstories.com/stories/a-strange-thing-to-talk-about-my-life-with-ocd/",
    "https://theocdstories.com/stories/the-blessing-of-accepting-uncertainty-in-ocd/",
    "https://theocdstories.com/stories/petrified-of-plagiarism/",
    "https://theocdstories.com/stories/on-avoiding-writing-this-essay/",
    "https://theocdstories.com/stories/beating-ocd-by-embracing-uncertainty/",
    "https://theocdstories.com/stories/emotional-abusive-relationship-with-somatosensory-ocd/",
    "https://theocdstories.com/stories/taming-the-beast-ocd-body-dysmorphia-and-depression/",
    "https://theocdstories.com/stories/have-hope-in-ocd-recovery/",
    "https://theocdstories.com/stories/university-rugby-and-sexual-orientation-ocd/",
    "https://theocdstories.com/stories/from-abuse-and-ocd-to-travelling-the-world-and-inspiring-others/",
    "https://theocdstories.com/stories/my-40-year-journey-with-ocd/",
    "https://theocdstories.com/stories/i-hate-you-an-ocd-obsession/",
    "https://theocdstories.com/stories/rational-and-irrational-parts/",
    "https://theocdstories.com/stories/february-fifteenth-my-obsession-with-obsession/",
    "https://theocdstories.com/stories/ocd-recovery-i-am-going-to-achieve-my-dreams/",
    "https://theocdstories.com/stories/whack-a-mole-brain-a-poem/",
    "https://theocdstories.com/stories/ocd-wanted-it-not-me/",
    "https://theocdstories.com/stories/the-ocd-comic/",
    "https://theocdstories.com/stories/maternal-ocd-and-facing-the-ocd-monster-head-on/",
    "https://theocdstories.com/stories/ocd-and-education/",
    "https://theocdstories.com/stories/standing-at-the-gate-of-the-journey/",
    "https://theocdstories.com/stories/my-ocd-transformation/",
    "https://theocdstories.com/stories/postpartum-ocd-and-faith/",
    "https://theocdstories.com/stories/erics-ocd-story-version-2/",
    "https://theocdstories.com/stories/my-naturopathic-ocd-recovery/",
    "https://theocdstories.com/stories/glimpses-of-a-life-without-ocd/",
    "https://theocdstories.com/stories/i-am-not-an-ocd-unicorn/",
    "https://theocdstories.com/stories/parenting-with-intrusive-thoughts/",
    "https://theocdstories.com/stories/my-ocd-story-experience-hope-and-recovery/",
    "https://theocdstories.com/stories/postpartum-ocd-intrusive-thoughts-and-recovery/",
    "https://theocdstories.com/stories/so-much-better-than-i-was/",
    "https://theocdstories.com/stories/a-true-paradox/",
    "https://theocdstories.com/stories/ocd-the-monster-in-my-mind/",
    "https://theocdstories.com/stories/my-obsession-the-fear-of-farting/",
    "https://theocdstories.com/stories/chasing-calm-my-life-with-pure-ocd/",
    "https://theocdstories.com/stories/exiting-the-maze-a-spiritual-answer-to-psychological-chaos/",
    "https://theocdstories.com/stories/my-road-to-recovery/",
    "https://theocdstories.com/stories/what-if-im-a-paedophile/",
    "https://theocdstories.com/stories/but-he-had-everything-to-live-for/",
    "https://theocdstories.com/stories/learning-to-live/",
    "https://theocdstories.com/stories/me-and-my-bully/",
    "https://theocdstories.com/stories/the-intruder/",
    "https://theocdstories.com/stories/recovering-from-ocd-one-day-at-a-time/",
    "https://theocdstories.com/stories/ocd-is-not-a-disease-that-bothers-its-a-disease-that-tortures/",
    "https://theocdstories.com/stories/the-fear-of-bad-things-and-the-optimism-of-recovery/",
    "https://theocdstories.com/stories/overcoming-ocd-and-ptsd/",
    "https://theocdstories.com/stories/ocd-and-emetophobia-gaining-my-life-back/",
    "https://theocdstories.com/stories/defeating-the-pain-of-ocd/",
    "https://theocdstories.com/stories/living-with-obsessive-compulsive-disorder/",
    "https://theocdstories.com/stories/ignorance-total-annihilation-two-odd-supports-for-ocd-recovery/",
    "https://theocdstories.com/stories/harm-ocd-religious-ocd-recovery-from-them-haleys-ocd-story/",
    "https://theocdstories.com/stories/pien-suffers-from-obsessive-compulsive-disorder/",
    "https://theocdstories.com/stories/from-contamination-ocd-to-the-bbc/",
    "https://theocdstories.com/stories/extracts-of-my-life/",
    "https://theocdstories.com/stories/fear-of-death-writing-the-alphabet-and-the-importance-of-embracing-ocd/",
    "https://theocdstories.com/stories/succeeding-despite-ocd/",
]


def slug_from_url(url):
    return url.rstrip("/").split("/")[-1]


def scrape_story(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title_el = soup.select_one("h1.entry-title, h1.post-title, h1")
    title = title_el.get_text(strip=True) if title_el else ""
    author_el = soup.select_one(".author-name, .entry-author-name, .post-author, a[rel='author']")
    author = author_el.get_text(strip=True) if author_el else "Anon"
    date_el = soup.select_one("time, .entry-date, .post-date")
    date = date_el.get_text(strip=True) if date_el else ""
    content_el = soup.select_one(".entry-content, .post-content, article .content")
    if content_el:
        for tag in content_el.select("script, style, .sharedaddy, .jp-relatedposts"):
            tag.decompose()
        paragraphs = content_el.find_all("p")
        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    else:
        text = ""
    return {"title": title, "author": author, "date": date, "text": text, "url": url}


def main():
    new_stories = []
    for i, url in enumerate(FAILED_URLS):
        slug = slug_from_url(url)
        if slug in scraped:
            continue
        print(f"[{i+1}/{len(FAILED_URLS)}] {slug}")
        try:
            story = scrape_story(url)
            story["slug"] = slug
            new_stories.append(story)
            txt_path = os.path.join(TEXTS_DIR, f"{slug}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(story["text"])
            time.sleep(2)
        except Exception as e:
            print(f"  ERROR: {e}")

    if new_stories:
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["slug", "title", "author", "date", "url", "word_count"])
            for s in new_stories:
                wc = len(s["text"].split())
                w.writerow({"slug": s["slug"], "title": s["title"], "author": s["author"],
                             "date": s["date"], "url": s["url"], "word_count": wc})

    total_new = len(new_stories)
    total_words = sum(len(s["text"].split()) for s in new_stories)
    print(f"\nRetry done: {total_new} new stories, {total_words} words")
    print(f"Total in corpus: {len(existing_rows) + total_new} stories")


if __name__ == "__main__":
    main()
