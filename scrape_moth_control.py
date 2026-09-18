"""
Scrape personal narratives from The Moth (themoth.org/stories)
as a control corpus for comparison with OCD narratives.
"""
import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import re

OUT_DIR = os.path.join(os.path.dirname(__file__), "control_moth")
TEXTS_DIR = os.path.join(OUT_DIR, "texts")
os.makedirs(TEXTS_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)"}
BASE = "https://themoth.org/stories"


def get_story_links(max_pages=20):
    links = []
    for page in range(1, max_pages + 1):
        url = f"{BASE}?page={page}" if page > 1 else BASE
        print(f"  Listing page {page}: {url}")
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"    HTTP {r.status_code}, stopping")
                break
            soup = BeautifulSoup(r.text, "html.parser")
            story_links = soup.select("a[href*='/stories/']")
            page_links = []
            for a in story_links:
                href = a.get("href", "")
                if "/stories/" in href and href != "/stories" and "?page=" not in href:
                    full = href if href.startswith("http") else f"https://themoth.org{href}"
                    page_links.append(full)
            page_links = list(dict.fromkeys(page_links))
            if not page_links:
                print("    No links found, stopping")
                break
            links.extend(page_links)
            print(f"    Found {len(page_links)} links")
            time.sleep(1.5)
        except Exception as e:
            print(f"    ERROR: {e}")
            break
    return list(dict.fromkeys(links))


def scrape_story(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.select_one("h1, .story-title, .title")
    title = title_el.get_text(strip=True) if title_el else ""

    author_el = soup.select_one(".storyteller-name, .author, .byline, a[href*='/storytellers/']")
    author = author_el.get_text(strip=True) if author_el else "Unknown"

    # The Moth has story transcripts in the main content area
    content_el = soup.select_one(".story-content, .entry-content, .post-content, article, main")
    if content_el:
        for tag in content_el.select("script, style, nav, footer, .share, .related"):
            tag.decompose()
        paragraphs = content_el.find_all("p")
        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    else:
        text = ""

    return {"title": title, "author": author, "text": text, "url": url}


def slug_from_url(url):
    parts = url.rstrip("/").split("/")
    return parts[-1] if parts else "unknown"


def main():
    print("Collecting story links from The Moth...")
    links = get_story_links(max_pages=25)
    print(f"Found {len(links)} unique story links\n")

    stories = []
    for i, url in enumerate(links):
        slug = slug_from_url(url)
        print(f"[{i+1}/{len(links)}] {slug}")
        try:
            story = scrape_story(url)
            if len(story["text"].split()) < 50:
                print(f"  Too short ({len(story['text'].split())} words), skipping")
                continue
            story["slug"] = slug
            stories.append(story)
            txt_path = os.path.join(TEXTS_DIR, f"{slug}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(story["text"])
            time.sleep(1.5)
        except Exception as e:
            print(f"  ERROR: {e}")

    csv_path = os.path.join(OUT_DIR, "metadata.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "title", "author", "url", "word_count"])
        w.writeheader()
        for s in stories:
            wc = len(s["text"].split())
            w.writerow({"slug": s["slug"], "title": s["title"], "author": s["author"],
                         "url": s["url"], "word_count": wc})

    total_words = sum(len(s["text"].split()) for s in stories)
    print(f"\nDone: {len(stories)} stories, {total_words} total words")
    print(f"Texts: {TEXTS_DIR}")
    print(f"Metadata: {csv_path}")


if __name__ == "__main__":
    main()
