"""
Build a control corpus from Project Gutenberg first-person autobiographies.
Downloads full texts, splits into chapter/segment-length passages
matching the OCD corpus length distribution.
"""
import requests
import os
import csv
import re
import statistics

HERE = os.path.dirname(__file__)
CONTROL_DIR = os.path.join(HERE, "control_gutenberg")
TEXTS_DIR = os.path.join(CONTROL_DIR, "texts")
os.makedirs(TEXTS_DIR, exist_ok=True)

# First-person autobiographies/memoirs on Gutenberg
BOOKS = [
    (148, "autobiography-benjamin-franklin"),
    (2376, "up-from-slavery-washington"),
    (23, "narrative-frederick-douglass"),
    (2397, "story-of-my-life-helen-keller"),
    (4390, "personal-memoirs-ulysses-grant-v1"),
    (4391, "personal-memoirs-ulysses-grant-v2"),
    (1636, "voyages-of-dr-dolittle"),  # first person adventure narrative
    (766, "david-copperfield-dickens"),  # first person literary
    (1400, "great-expectations-dickens"),  # first person literary
    (514, "little-women-alcott"),  # mix of perspectives, personal
    (45, "anne-of-green-gables"),  # third person but personal narrative
    (74, "adventures-of-tom-sawyer"),  # third person, young narrator
    (76, "adventures-huckleberry-finn"),  # first person
    (174, "picture-of-dorian-gray"),  # third person literary
    (11, "alices-adventures-in-wonderland"),  # third person
    (16, "peter-pan"),  # third person
    (1260, "jane-eyre-bronte"),  # first person
    (345, "dracula-stoker"),  # first person journal entries
    (25344, "scarlet-pimpernel"),  # third person
    (120, "treasure-island"),  # first person
]


def download_gutenberg(ebook_id):
    urls = [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 1000:
                return r.text
        except Exception:
            continue
    return None


def strip_gutenberg_header_footer(text):
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "E-text prepared by",
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            nl = text.find("\n", idx)
            if nl != -1:
                start_idx = nl + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def split_into_segments(text, target_words=1400, min_words=300, max_words=5000):
    """Split text into segments of approximately target_words length.
    Uses chapter boundaries when available, otherwise paragraph groups."""

    # Try splitting by chapters first
    chapter_pattern = re.compile(
        r'\n\s*(CHAPTER|Chapter|PART|Part)\s+[IVXLCDM\d]+[.\s]*\n',
        re.MULTILINE
    )
    chapters = chapter_pattern.split(text)

    segments = []
    if len(chapters) > 3:
        current = ""
        for chunk in chapters:
            chunk = chunk.strip()
            if not chunk:
                continue
            words = len(chunk.split())
            if words < min_words and current:
                current += "\n\n" + chunk
            elif words > max_words:
                # Split large chapters into paragraphs
                paras = chunk.split("\n\n")
                sub = ""
                for p in paras:
                    p = p.strip()
                    if not p:
                        continue
                    if len((sub + " " + p).split()) > target_words * 1.3 and len(sub.split()) >= min_words:
                        segments.append(sub.strip())
                        sub = p
                    else:
                        sub += "\n\n" + p if sub else p
                if len(sub.split()) >= min_words:
                    segments.append(sub.strip())
            else:
                if current and len((current + " " + chunk).split()) > target_words * 1.5:
                    if len(current.split()) >= min_words:
                        segments.append(current.strip())
                    current = chunk
                else:
                    current += "\n\n" + chunk if current else chunk

        if current and len(current.split()) >= min_words:
            segments.append(current.strip())
    else:
        # No clear chapter structure - split by paragraph groups
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        for p in paras:
            candidate = current + "\n\n" + p if current else p
            if len(candidate.split()) > target_words * 1.3 and len(current.split()) >= min_words:
                segments.append(current.strip())
                current = p
            else:
                current = candidate
        if current and len(current.split()) >= min_words:
            segments.append(current.strip())

    return segments


def main():
    # Get OCD corpus stats for matching
    ocd_meta = os.path.join(HERE, "metadata.csv")
    ocd_wcs = []
    with open(ocd_meta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ocd_wcs.append(int(row["word_count"]))
    target_words = int(statistics.median(ocd_wcs))
    print(f"OCD corpus median word count: {target_words}")
    print(f"OCD corpus size: {len(ocd_wcs)} stories\n")

    all_segments = []
    for ebook_id, slug in BOOKS:
        print(f"Downloading {slug} (ebook {ebook_id})...")
        raw = download_gutenberg(ebook_id)
        if raw is None:
            print(f"  FAILED to download")
            continue

        text = strip_gutenberg_header_footer(raw)
        total_words = len(text.split())
        print(f"  {total_words:,} words")

        segments = split_into_segments(text, target_words=target_words)
        print(f"  Split into {len(segments)} segments")

        for i, seg in enumerate(segments):
            seg_slug = f"{slug}-seg{i+1:03d}"
            wc = len(seg.split())
            all_segments.append({
                "slug": seg_slug,
                "source": slug,
                "ebook_id": ebook_id,
                "segment": i + 1,
                "word_count": wc,
                "text": seg,
            })

    print(f"\nTotal segments: {len(all_segments)}")

    # Save texts
    for seg in all_segments:
        txt_path = os.path.join(TEXTS_DIR, f"{seg['slug']}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(seg["text"])

    # Save metadata
    csv_path = os.path.join(CONTROL_DIR, "metadata.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "source", "ebook_id", "segment", "word_count"])
        w.writeheader()
        for seg in all_segments:
            w.writerow({k: seg[k] for k in ["slug", "source", "ebook_id", "segment", "word_count"]})

    wcs = [s["word_count"] for s in all_segments]
    print(f"\nControl corpus: {len(all_segments)} segments")
    print(f"Total words: {sum(wcs):,}")
    print(f"Mean: {statistics.mean(wcs):.0f}, Median: {statistics.median(wcs):.0f}")
    print(f"Min: {min(wcs)}, Max: {max(wcs)}")


if __name__ == "__main__":
    main()
