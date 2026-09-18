"""
Scrape patient cancer stories from thepatientstory.com as control corpus.
Collects first-person illness narratives across all cancer types.
"""
import os
import csv
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "control_patient_stories")
TEXTS_DIR = os.path.join(OUT_DIR, "texts")
os.makedirs(TEXTS_DIR, exist_ok=True)

BASE = "https://www.thepatientstory.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY = 1.5

CATEGORIES = [
    "anal-cancer", "bladder-cancer", "breast-cancer", "cervical-cancer",
    "colorectal-cancer", "hodgkin-lymphoma", "kidney-cancer", "leukemia",
    "lung-cancer", "multiple-myeloma", "mpn", "non-hodgkin-lymphoma",
    "ovarian-cancer", "prostate-cancer", "sarcoma", "skin-cancer",
    "stomach-cancer", "testicular-cancer", "thyroid-cancer",
]


def fetch(url):
    time.sleep(DELAY)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


SKIP_SLUGS = {
    "bladder-cancer-basics", "causes-symptoms-2", "diagnosis-and-treatment",
    "black-patient-series", "treatments", "chemotherapy", "radiation",
    "immunotherapy", "surgery", "clinical-trials", "support", "resources",
    "about", "contact", "donate", "newsletter", "search", "privacy-policy",
}

INFO_PATTERNS = [
    "basics", "causes", "symptoms", "treatment", "diagnosis", "screening",
    "prevention", "risk-factors", "staging", "prognosis", "statistics",
    "overview", "types", "side-effects", "support-groups", "faq",
]


def is_story_url(href):
    """Check if URL points to an individual story (has patient name as last segment)."""
    if not href or "patient-stories" not in href:
        return False
    if "#" in href or "?" in href:
        return False
    parts = [p for p in href.split("/") if p]
    if len(parts) < 4:
        return False
    last = parts[-1]
    if last.startswith("page") or last in CATEGORIES:
        return False
    if last in SKIP_SLUGS:
        return False
    if any(last.endswith(x) for x in ["-cancer", "-lymphoma", "-myeloma", "-leukemia"]):
        return False
    if any(pat in last for pat in INFO_PATTERNS):
        return False
    # Must contain a hyphen (person names like "amanda-r", "john-m")
    if "-" not in last:
        return False
    return True


def collect_story_urls_from_page(soup, base_url):
    """Extract individual story URLs from a listing page."""
    urls = set()
    for a in soup.select("a[href]"):
        href = a["href"]
        if not href.startswith("http"):
            href = urljoin(base_url, href)
        if is_story_url(href):
            urls.add(href.rstrip("/"))
    return urls


def get_pagination_urls(soup, base_url):
    """Find pagination links (page/2/, page/3/, etc.)."""
    pages = set()
    for a in soup.select("a[href]"):
        href = a["href"]
        if "/page/" in href:
            if not href.startswith("http"):
                href = urljoin(base_url, href)
            pages.add(href.rstrip("/"))
    return pages


def collect_subcategory_urls(soup, category):
    """Find subcategory listing pages within a cancer type."""
    subs = set()
    for a in soup.select("a[href]"):
        href = a["href"]
        if not href.startswith("http"):
            href = urljoin(BASE, href)
        if f"/patient-stories/{category}/" in href:
            parts = [p for p in href.split("/") if p]
            # Subcategory has one more level than category
            if len(parts) >= 4 and not is_story_url(href):
                subs.add(href.rstrip("/"))
    return subs


def crawl_listing_page(url, visited_pages):
    """Crawl a listing page and its pagination, collecting story URLs."""
    if url in visited_pages:
        return set()
    visited_pages.add(url)

    try:
        soup = fetch(url)
    except Exception as e:
        print(f"    Error fetching {url}: {e}")
        return set()

    stories = collect_story_urls_from_page(soup, url)
    pagination = get_pagination_urls(soup, url)

    for page_url in sorted(pagination):
        if page_url not in visited_pages:
            visited_pages.add(page_url)
            try:
                page_soup = fetch(page_url)
                stories |= collect_story_urls_from_page(page_soup, page_url)
            except Exception as e:
                print(f"    Error fetching page {page_url}: {e}")

    return stories


def scrape_story(url):
    """Scrape an individual patient story page."""
    try:
        soup = fetch(url)
    except Exception as e:
        return None

    # Extract title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Extract main content — story text is in .entry-content or article
    content_div = soup.select_one(".entry-content") or soup.select_one("article")
    if not content_div:
        return None

    # Remove navigation elements, sidebars, related posts
    for tag in content_div.select("nav, .sidebar, .related, .wp-block-group, .tps-toc, script, style, .sharedaddy"):
        tag.decompose()

    # Extract paragraphs and headings
    text_parts = []
    for el in content_div.find_all(["p", "h2", "h3", "h4", "blockquote"]):
        t = el.get_text(strip=True)
        if t and len(t) > 10:
            # Skip boilerplate
            if any(skip in t.lower() for skip in [
                "subscribe", "sign up", "newsletter", "cookie", "privacy policy",
                "sponsored by", "disclaimer", "copyright", "all rights reserved",
                "table of contents", "on this page"
            ]):
                continue
            text_parts.append(t)

    text = "\n\n".join(text_parts)

    # Extract patient name from URL
    slug = url.rstrip("/").split("/")[-1]

    # Extract cancer type from URL
    parts = url.split("/patient-stories/")
    cancer_type = ""
    if len(parts) > 1:
        cancer_type = parts[1].split("/")[0]

    return {
        "slug": slug,
        "title": title,
        "cancer_type": cancer_type,
        "url": url,
        "text": text,
        "word_count": len(text.split()),
    }


def main():
    print("=" * 60)
    print("Scraping The Patient Story (thepatientstory.com)")
    print("=" * 60)

    # Phase 1: Collect all story URLs
    all_story_urls = set()
    visited_pages = set()

    for cat in CATEGORIES:
        cat_url = f"{BASE}/patient-stories/{cat}/"
        print(f"\n[{cat}] Crawling category...")

        try:
            cat_soup = fetch(cat_url)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # Get stories directly on category page
        stories = collect_story_urls_from_page(cat_soup, cat_url)
        pagination = get_pagination_urls(cat_soup, cat_url)

        # Follow pagination on category page
        for page_url in sorted(pagination):
            stories |= crawl_listing_page(page_url, visited_pages)

        # Get subcategory pages
        subcats = collect_subcategory_urls(cat_soup, cat)
        print(f"  Found {len(subcats)} subcategories, {len(stories)} stories on main page")

        for sub_url in sorted(subcats):
            sub_stories = crawl_listing_page(sub_url, visited_pages)
            stories |= sub_stories

        all_story_urls |= stories
        print(f"  Total for {cat}: {len(stories)} stories")

    print(f"\n{'=' * 60}")
    print(f"Total unique story URLs found: {len(all_story_urls)}")
    print(f"{'=' * 60}")

    # Phase 2: Scrape each story
    metadata = []
    existing = set()
    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(row["slug"])
                metadata.append(row)
        print(f"Resuming: {len(existing)} stories already scraped")

    def safe_slug(url):
        s = url.rstrip("/").split("/")[-1]
        s = re.sub(r'[^\w\-]', '', s)
        return s

    new_urls = [u for u in sorted(all_story_urls)
                if safe_slug(u) not in existing and safe_slug(u)]
    print(f"New stories to scrape: {len(new_urls)}\n")

    for i, url in enumerate(new_urls):
        slug = safe_slug(url)
        if not slug:
            continue
        print(f"[{i+1}/{len(new_urls)}] {slug}...")

        result = scrape_story(url)
        if result is None or result["word_count"] < 100:
            print(f"  Skipped (no content or too short)")
            continue

        result["slug"] = slug
        txt_path = os.path.join(TEXTS_DIR, f"{slug}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        meta_row = {
            "slug": result["slug"],
            "title": result["title"],
            "cancer_type": result["cancer_type"],
            "url": result["url"],
            "word_count": result["word_count"],
        }
        metadata.append(meta_row)

        # Save metadata incrementally
        if (i + 1) % 10 == 0 or i == len(new_urls) - 1:
            with open(meta_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["slug", "title", "cancer_type", "url", "word_count"])
                w.writeheader()
                w.writerows(metadata)

        print(f"  OK ({result['word_count']} words)")

    # Final save
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "title", "cancer_type", "url", "word_count"])
        w.writeheader()
        w.writerows(metadata)

    print(f"\n{'=' * 60}")
    print(f"DONE: {len(metadata)} stories saved")
    print(f"Texts in: {TEXTS_DIR}")
    print(f"Metadata: {meta_path}")

    # Summary stats
    wcs = [int(m["word_count"]) for m in metadata]
    if wcs:
        import numpy as np
        wcs = np.array(wcs)
        print(f"\nWord count stats:")
        print(f"  Mean: {wcs.mean():.0f}")
        print(f"  Median: {np.median(wcs):.0f}")
        print(f"  Min: {wcs.min()}, Max: {wcs.max()}")
        print(f"  Total words: {wcs.sum():,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
