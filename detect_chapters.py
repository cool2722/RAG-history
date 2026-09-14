#!/usr/bin/env python3
"""
Detect chapter boundaries in the NCERT history corpus.

WHY THIS EXISTS
---------------
The corpus arrived at two different granularities (see download_ncert.py):
  - NCERT's own ZIPs (kehs*, lehs*)  -> already ONE FILE PER CHAPTER.
  - Vajiram whole-book PDFs           -> ONE FILE PER ENTIRE BOOK.

So "one file" does not mean "one chapter" consistently. Without this step you
get chapter-level metadata for roughly half the corpus and only book-level for
the rest, which makes gold chunk IDs for multi-hop eval questions much harder
to pin down.

WHAT IT ACTUALLY FOUND (verified against the real files, not assumed)
---------------------------------------------------------------------
There is no single heading convention. Each book differs:

  class6_our_pasts_1   -> "CHAPTER 1" on its own line, ALL CAPS. 12 markers.
  class7_our_pasts_2   -> NO chapter markers in the body at all. Has a clean
                          table of contents ("1. Tracing Changes Through A
                          Thousand Years") -- titles are matched into the body.
  class8_our_pasts_3   -> NO chapter markers, AND ~14k characters encoded in the
                          Unicode Private Use Area (U+F000-U+F0FF) because the
                          source PDF used a non-standard font mapping. Chapter
                          headings are literally unreadable until decoded with a
                          -0xF000 offset. This script decodes them.
  class9_india_...     -> "Chapter I" style (Roman numerals, title case), but
                          OCR-damaged: "Chapter I" repeats 5x in a TOC block,
                          "Chapter II" is missing, and one reads
                          "Chapter VIIIIIIIIII" instead of "Chapter VII".
  class10_india_...    -> "Chapter I".."Chapter VIII", clean and sequential.

The kehs*/lehs* files are left alone: they are already one chapter per file, so
their chapter identity comes from the filename.

OUTPUT
------
  chapter_index.csv  -- one row per detected chapter:
      book, chapter_num, chapter_title, start_line, end_line,
      start_page, detection_method, confidence

  Optionally (--split) writes each chapter to its own .txt under
  ncert_history_corpus_chapters/<book>/<NN>_<slug>.txt

Confidence is reported honestly per chapter:
  high   = explicit marker matched, sequential numbering intact
  medium = matched via table-of-contents title lookup
  low    = marker found but numbering is damaged/out of sequence -> CHECK BY HAND

USAGE
-----
  python detect_chapters.py                 # write chapter_index.csv
  python detect_chapters.py --split         # also split into per-chapter files
  python detect_chapters.py --input DIR     # override input dir
"""

import os
import re
import csv
import sys
import argparse

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DEFAULT_INPUT = "ncert_history_corpus_text"
CHAPTER_OUTPUT_DIR = "ncert_history_corpus_chapters"
INDEX_CSV = "chapter_index.csv"

# Files that are already one-chapter-per-file (from NCERT's ZIPs).
# Matched by filename prefix.
ALREADY_CHAPTERED_PREFIXES = ("kehs", "lehs")

# Roman numeral -> int, for the "Chapter IV" style books.
ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}


def decode_pua(text: str) -> str:
    """
    Decode Private Use Area characters (U+F000-U+F0FF) back to ASCII.

    class8_our_pasts_3.txt has ~14,000 of these because its source PDF used a
    non-standard font encoding. Without this, its chapter headings are literally
    invisible to any search -- they extract as \\uf057\\uf065... rather than
    "Weavers, Iron Smelters and Factory Owners".
    """
    return "".join(
        chr(ord(c) - 0xF000) if 0xF000 <= ord(c) <= 0xF0FF else c
        for c in text
    )


def normalise_roman(token: str):
    """
    Map a roman-numeral token to an int, tolerating OCR repetition damage
    like 'VIIIIIIIIII'. Returns (value, was_repaired) or (None, False).
    """
    token = token.upper().strip()
    if token in ROMAN:
        return ROMAN[token], False
    # OCR sometimes multiplies a trailing letter: VIIIIIIIIII -> VII
    collapsed = re.sub(r"(I{2,})", lambda m: "I" * min(len(m.group(1)), 3), token)
    if collapsed in ROMAN:
        return ROMAN[collapsed], True
    return None, False


def find_page_for_line(lines, idx: int):
    """Walk backwards to the most recent '--- PAGE n ---' marker."""
    for j in range(idx, -1, -1):
        m = re.match(r"---\s*PAGE\s+(\d+)\s*---", lines[j].strip())
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------
# Strategy 1: explicit "CHAPTER 1" / "Chapter IV" markers on their own line
# ---------------------------------------------------------------------
MARKER_RE = re.compile(r"^\s*(?:CHAPTER|Chapter|CHAPTER:)\s+([IVXLC]+|\d{1,2})\s*$")


def detect_by_marker(lines):
    """Return list of (line_idx, chapter_num, raw_token, repaired_flag)."""
    found = []
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m:
            continue
        token = m.group(1)
        if token.isdigit():
            num, repaired = int(token), False
        else:
            num, repaired = normalise_roman(token)
            if num is None:
                continue
        found.append((i, num, token, repaired))
    return found


def dedupe_markers(found):
    """
    Drop consecutive duplicates of the same chapter number that sit within a
    few lines of each other -- these are table-of-contents blocks, not real
    chapter starts (class9 repeats 'Chapter I' five times in a row).
    Keeps the LAST occurrence of such a run, which is the real body heading.
    """
    if not found:
        return []
    cleaned = []
    for entry in found:
        if cleaned and entry[1] == cleaned[-1][1] and entry[0] - cleaned[-1][0] <= 10:
            cleaned[-1] = entry  # replace with later occurrence
        else:
            cleaned.append(entry)
    return cleaned


# ---------------------------------------------------------------------
# Strategy 2: bare chapter number followed by an ALL-CAPS title
# ---------------------------------------------------------------------
# class7/class8 print the chapter number alone on a line, then the title in
# caps (possibly wrapped over two lines):
#       2
#       NEW KINGS AND
#       KINGDOMS
#
# The trap: even-numbered PAGE numbers are also bare numbers, and they are
# followed by the book's running header ("OUR PASTS - II"), which is also all
# caps. So we must reject any candidate whose "title" is the running header.

def find_running_header(lines, sample=4000):
    """
    Identify BOOK-WIDE running furniture (e.g. "OUR PASTS - II", "KEYWORDS")
    that must not be mistaken for a chapter title.

    Important subtlety, learned from class7: a chapter's own title is often
    ALSO repeated as a running header on every page OF THAT CHAPTER. So
    "appears many times" is NOT sufficient grounds for rejection -- that would
    throw away the real titles.

    The distinguishing feature is SPREAD, not count:
      - book-wide furniture appears across the whole file (first page to last)
      - a chapter title's repetitions cluster in one contiguous band
    So we reject only lines whose occurrences span most of the document.
    """
    from collections import defaultdict
    positions = defaultdict(list)
    limit = min(len(lines), sample) if sample else len(lines)
    for i, l in enumerate(lines[:limit]):
        s = l.strip()
        if 4 <= len(s) <= 40 and s == s.upper() and re.search(r"[A-Z]{3}", s):
            positions[s].append(i)

    if not positions:
        return set()

    doc_len = max(1, limit)
    headers = set()
    for s, pos in positions.items():
        if len(pos) < 3:
            continue
        spread = (pos[-1] - pos[0]) / doc_len
        # Spans more than half the sampled document -> book-level furniture.
        if spread > 0.5:
            headers.add(s)
    return headers


NUM_ONLY_RE = re.compile(r"^\s*(\d{1,2})\s*$")


def detect_by_numbered_caps(lines):
    """
    Return list of (line_idx, chapter_num, title).
    Requires the number to be followed by an ALL-CAPS line that is NOT the
    book's running header.
    """
    headers = find_running_header(lines)
    found = []
    for i, line in enumerate(lines):
        m = NUM_ONLY_RE.match(line)
        if not m:
            continue
        num = int(m.group(1))
        if not (1 <= num <= 20):
            continue

        nxt = []
        for j in range(i + 1, min(i + 5, len(lines))):
            c = lines[j].strip()
            if c:
                nxt.append(c)
            if len(nxt) >= 3:
                break
        if not nxt:
            continue

        first = nxt[0]
        if first in headers:
            continue  # running header, not a chapter title
        if not (len(first) >= 4 and first == first.upper()
                and re.search(r"[A-Z]{3}", first)):
            continue

        # Title may wrap onto the next line, also in caps and also not a header.
        title = first
        if len(nxt) > 1:
            second = nxt[1]
            if (second == second.upper() and re.search(r"[A-Z]{3}", second)
                    and second not in headers and len(second) <= 40):
                title = f"{first} {second}"

        found.append((i, num, title.strip()))

    # Keep only the first occurrence of each chapter number, in order.
    seen = {}
    for idx, num, title in found:
        if num not in seen:
            seen[num] = (idx, num, title)
    ordered = sorted(seen.values())

    # Sanity pass. Two failure modes seen in real data (class7):
    #  1. the same title attached to two different chapter numbers
    #     (a bare page number happened to precede a running header)
    #  2. a title accidentally doubled ("THE MUGHAL EMPIRE THE MUGHAL EMPIRE")
    #     when the wrap-join picked up the header repeated on the next line
    cleaned = []
    used_titles = set()
    for idx, num, title in ordered:
        # de-duplicate an immediately repeated phrase
        half = len(title) // 2
        if len(title) > 10 and title[:half].strip() == title[half:].strip():
            title = title[:half].strip()

        key = normalise_title(title)
        if key in used_titles:
            # Same title already claimed by an earlier chapter -> this one is
            # a false positive from a page number + running header.
            continue
        used_titles.add(key)
        cleaned.append((idx, num, title))

    # Chapter numbers should increase with position in the document; if they
    # do not, the numbering is unreliable, so renumber positionally and let
    # the caller mark it lower confidence.
    cleaned.sort()
    nums = [c[1] for c in cleaned]
    if nums != sorted(nums):
        cleaned = [(idx, k + 1, title) for k, (idx, _, title) in enumerate(cleaned)]

    return cleaned


def detect_by_title_bands(lines, min_chapters=3):
    """
    Fallback for books that print no chapter number at the chapter start
    (class8): each chapter's title is repeated as a running header on every
    page of that chapter, so the title's occurrences form a contiguous BAND.
    The chapter therefore starts at (or just before) the first occurrence.

    We take every ALL-CAPS line that repeats 3+ times, drop book-wide
    furniture, then order the surviving titles by first appearance.
    """
    from collections import defaultdict
    positions = defaultdict(list)
    for i, l in enumerate(lines):
        s = l.strip()
        if 6 <= len(s) <= 70 and s == s.upper() and re.search(r"[A-Z]{4}", s):
            positions[s].append(i)

    doc_len = max(1, len(lines))
    bands = []
    for s, pos in positions.items():
        if len(pos) < 3:
            continue
        spread = (pos[-1] - pos[0]) / doc_len
        if spread > 0.5:
            continue  # book-wide furniture (e.g. "OUR PASTS - III")
        if re.fullmatch(r"[\d\s\W]+", s):
            continue
        bands.append((pos[0], s, len(pos)))

    bands.sort()
    # Drop bands that begin inside the front matter / contents region.
    bands = [b for b in bands if b[0] > 260]

    if len(bands) < min_chapters:
        return []

    return [(start, k + 1, title) for k, (start, title, _) in enumerate(bands)]


# ---------------------------------------------------------------------
# Strategy 4: table-of-contents title matching (last resort)
# ---------------------------------------------------------------------
TOC_RE = re.compile(r"^\s*(\d{1,2})\.\s+(.{6,80}?)\s*$")


def extract_toc(lines, scan_limit=600):
    """
    Pull '1. Tracing Changes Through A Thousand Years' style entries from the
    front matter. Returns {chapter_num: title}.
    """
    toc = {}
    for i, line in enumerate(lines[:scan_limit]):
        m = TOC_RE.match(line)
        if not m:
            continue
        num = int(m.group(1))
        title = m.group(2).strip()
        # A TOC entry is followed (within a line or two) by a page number.
        lookahead = " ".join(l.strip() for l in lines[i + 1:i + 3])
        if not re.match(r"^\d{1,3}\b", lookahead):
            continue
        if 1 <= num <= 20 and num not in toc:
            toc[num] = title
    return toc


def normalise_title(s: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy title comparison."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def detect_by_toc(lines, toc, search_start=0):
    """
    For each TOC title, find where it reappears in the body as a heading.
    Returns list of (line_idx, chapter_num, title).
    """
    found = []
    norm_lines = [normalise_title(l) for l in lines]
    for num, title in sorted(toc.items()):
        target = normalise_title(title)
        if not target:
            continue
        best = None
        for i in range(search_start, len(lines)):
            nl = norm_lines[i]
            if not nl:
                continue
            # Heading lines are short; body prose that merely mentions the
            # phrase is usually much longer.
            if target in nl and len(nl) <= len(target) + 25:
                best = i
                break
        if best is not None:
            found.append((best, num, title))
    found.sort()
    return found


# ---------------------------------------------------------------------
# Main per-file detection
# ---------------------------------------------------------------------
def detect_chapters_in_file(path):
    """
    Returns (rows, method) where rows is a list of dicts describing chapters.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    had_pua = any(0xF000 <= ord(c) <= 0xF0FF for c in raw)
    if had_pua:
        raw = decode_pua(raw)

    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    book = os.path.splitext(os.path.basename(path))[0]

    # --- try explicit markers first ---
    markers = dedupe_markers(detect_by_marker(lines))

    rows = []
    method = None

    if len(markers) >= 3:
        method = "marker"
        nums = [m[1] for m in markers]
        sequential = nums == sorted(nums) and len(set(nums)) == len(nums)
        expected = list(range(min(nums), min(nums) + len(nums)))
        gap_free = nums == expected

        # Junk that appears right after a marker when the "marker" is really a
        # running footer at the foot of a page rather than a chapter heading.
        JUNK = re.compile(
            r"^(\d{4}-\d{2}|\d{1,4}|[IVXLC]+|---\s*PAGE.*)$", re.I
        )

        titles_are_junk = 0
        for k, (idx, num, token, repaired) in enumerate(markers):
            end = markers[k + 1][0] - 1 if k + 1 < len(markers) else len(lines) - 1
            title = ""
            for j in range(idx + 1, min(idx + 6, len(lines))):
                cand = lines[j].strip()
                if cand and not JUNK.match(cand):
                    title = cand
                    break
            if not title:
                titles_are_junk += 1

            if repaired:
                conf = "low"
            elif sequential and gap_free:
                conf = "high"
            else:
                conf = "low"

            rows.append({
                "book": book,
                "chapter_num": num,
                "chapter_title": title,
                "start_line": idx,
                "end_line": end,
                "start_page": find_page_for_line(lines, idx),
                "detection_method": "marker" + ("+roman_repair" if repaired else ""),
                "confidence": conf,
                "pua_decoded": had_pua,
            })

        # If most "chapters" have no real title after them, the markers are
        # almost certainly running page footers, not chapter headings.
        # class9 does exactly this: "Chapter III" sits at the foot of every
        # page of chapter 3, and real content flows straight past it.
        if titles_are_junk >= max(2, len(markers) // 2):
            for r in rows:
                r["confidence"] = "low"
                r["detection_method"] = "marker_but_looks_like_running_footer"
        return rows, method, lines

    # --- bare number + ALL CAPS title (class7 / class8 style) ---
    numbered = detect_by_numbered_caps(lines)
    if len(numbered) >= 3:
        method = "numbered_caps"
        for k, (idx, num, title) in enumerate(numbered):
            end = numbered[k + 1][0] - 1 if k + 1 < len(numbered) else len(lines) - 1
            rows.append({
                "book": book,
                "chapter_num": num,
                "chapter_title": title,
                "start_line": idx,
                "end_line": end,
                "start_page": find_page_for_line(lines, idx),
                "detection_method": "numbered_caps",
                "confidence": "high" if title else "low",
                "pua_decoded": had_pua,
            })
        return rows, method, lines

    # --- title-band fallback (class8 style: no chapter numbers at all) ---
    bands = detect_by_title_bands(lines)
    if len(bands) >= 3:
        method = "title_bands"
        for k, (idx, num, title) in enumerate(bands):
            end = bands[k + 1][0] - 1 if k + 1 < len(bands) else len(lines) - 1
            rows.append({
                "book": book,
                "chapter_num": num,
                "chapter_title": title,
                "start_line": idx,
                "end_line": end,
                "start_page": find_page_for_line(lines, idx),
                "detection_method": "title_band",
                # Numbering here is positional (order of appearance), not read
                # from the page, so it may not match the printed chapter number.
                "confidence": "medium",
                "pua_decoded": had_pua,
            })
        return rows, method, lines

    # --- fall back to table-of-contents title matching ---
    toc = extract_toc(lines)
    if toc:
        # Start searching past the TOC region itself.
        toc_end = 0
        for i, line in enumerate(lines[:600]):
            if TOC_RE.match(line):
                toc_end = i
        hits = detect_by_toc(lines, toc, search_start=toc_end + 1)
        if len(hits) >= 3:
            method = "toc"
            for k, (idx, num, title) in enumerate(hits):
                end = hits[k + 1][0] - 1 if k + 1 < len(hits) else len(lines) - 1
                rows.append({
                    "book": book,
                    "chapter_num": num,
                    "chapter_title": title,
                    "start_line": idx,
                    "end_line": end,
                    "start_page": find_page_for_line(lines, idx),
                    "detection_method": "toc_title_match",
                    "confidence": "medium",
                    "pua_decoded": had_pua,
                })
            return rows, method, lines

    return [], None, lines


def slugify(s, maxlen=48):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return (s[:maxlen] or "untitled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"directory of extracted .txt files (default: {DEFAULT_INPUT})")
    ap.add_argument("--split", action="store_true",
                    help="also write one .txt per detected chapter")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: input dir '{args.input}' not found.")
        print("Run this from the folder containing your extracted text, "
              "or pass --input /path/to/dir")
        sys.exit(1)

    # Gather .txt files (walk, since extract_text.py nests them per book folder)
    txt_files = []
    for dirpath, _, filenames in os.walk(args.input):
        for fn in sorted(filenames):
            if fn.lower().endswith(".txt"):
                txt_files.append(os.path.join(dirpath, fn))

    if not txt_files:
        print(f"No .txt files found under {args.input}")
        sys.exit(1)

    all_rows = []
    skipped = []

    for path in sorted(txt_files):
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]

        if stem.startswith(ALREADY_CHAPTERED_PREFIXES):
            skipped.append(stem)
            continue

        rows, method, lines = detect_chapters_in_file(path)

        if not rows:
            print(f"{stem}: NO CHAPTERS DETECTED -- inspect this file by hand.")
            all_rows.append({
                "book": stem, "chapter_num": "", "chapter_title": "",
                "start_line": "", "end_line": "", "start_page": "",
                "detection_method": "none", "confidence": "FAILED",
                "pua_decoded": "",
            })
            continue

        lows = sum(1 for r in rows if r["confidence"] == "low")
        note = f" ({lows} low-confidence -- check by hand)" if lows else ""
        pua = " [PUA decoded]" if rows[0].get("pua_decoded") else ""
        print(f"{stem}: {len(rows)} chapters via {method}{note}{pua}")

        all_rows.extend(rows)

        if args.split:
            out_dir = os.path.join(CHAPTER_OUTPUT_DIR, stem)
            os.makedirs(out_dir, exist_ok=True)
            for r in rows:
                body = "\n".join(lines[r["start_line"]:r["end_line"] + 1])
                fname = f"{int(r['chapter_num']):02d}_{slugify(r['chapter_title'])}.txt"
                with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                    f.write(body)

    if skipped:
        print(f"\nSkipped {len(skipped)} already-one-chapter-per-file sources "
              f"(kehs*/lehs*) -- their chapter identity comes from the filename.")

    fieldnames = ["book", "chapter_num", "chapter_title", "start_line",
                  "end_line", "start_page", "detection_method",
                  "confidence", "pua_decoded"]
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    detected = [r for r in all_rows if r["confidence"] != "FAILED"]
    low = [r for r in detected if r["confidence"] == "low"]
    print(f"\nWrote {INDEX_CSV}: {len(detected)} chapters across "
          f"{len(set(r['book'] for r in detected))} books.")
    if low:
        print(f"{len(low)} chapter(s) marked LOW confidence -- verify these "
              f"start_line values by hand before trusting the metadata:")
        for r in low:
            print(f"   {r['book']} ch{r['chapter_num']} @line {r['start_line']} "
                  f": {r['chapter_title'][:50]}")
    if args.split:
        print(f"Per-chapter files written under {CHAPTER_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()