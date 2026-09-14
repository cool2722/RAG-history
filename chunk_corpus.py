#!/usr/bin/env python3
"""
Chunk the NCERT history corpus for embedding.

WHY SENTENCE-AWARE FIXED-SIZE AND NOT THE USUAL DEFAULT
--------------------------------------------------------
The standard recommendation is recursive character splitting on
"\\n\\n" -> "\\n" -> sentence -> char. Measured against this corpus, that is
the wrong choice:

  file                                   avg line length   blank lines not
                                                            near a page marker
  kehs103                                      38                  0
  lehs203                                      33                  1
  class6_our_pasts_1                           33                  1
  class10_india_and_contemporary_world_2       48                 17

Average line length of 33-48 characters means the text is hard-wrapped at PDF
column width, and blank lines occur almost only around "--- PAGE n ---".
So splitting on "\\n\\n" degenerates into page-boundary splitting, and
splitting on "\\n" cuts every six words mid-sentence.

This script therefore:
  1. DE-WRAPS hard-wrapped lines back into sentences (the step that makes
     everything else work),
  2. tags non-prose furniture instead of blindly mixing or deleting it,
  3. packs sentences to a target token budget with overlap,
  4. uses the author's own numbered sections where they exist.

FURNITURE
---------
Textbook pages interleave prose with captions, glossary boxes, source
extracts and exercises. Measured density:

  class10 : 184 figure captions, 49 new-words boxes, 27 source boxes
  class6  : 150 timeline rows, 41 exercise prompts
  lehs101 :  30 figure captions,  3 source boxes

These are NOT deleted. Eval question E10's answer lives in a Class 7 glossary
box, so deleting furniture would delete a gold answer. Instead each chunk
carries a content_type, so retrieval can filter or boost, and you can report
results per content type.

SECTIONS
--------
The Class 11/12 files carry the author's own numbered sections
("1. Beginnings", "2. Subsistence Strategies") -- 8 to 12 per chapter. Where
present these are used as hard chunk boundaries, which keeps semantic units
intact. Files without them (kehs103 has none, the OCR'd books have none) fall
back to pure sentence packing. The metadata records which path was taken so
you can measure whether section awareness actually helped.

OUTPUT
------
  chunks.jsonl   one JSON object per chunk:
      chunk_id, book, chapter_num, chapter_title, section,
      page_start, page_end, content_type, source_type,
      n_tokens_est, text

  chunk_stats.csv  per-book counts, so you can sanity-check the distribution
                   before spending time on embeddings.

USAGE
-----
  python chunk_corpus.py
  python chunk_corpus.py --target-tokens 400 --overlap 0.15
  python chunk_corpus.py --index chapter_index_repaired.csv
"""

import os
import re
import csv
import json
import argparse
from collections import Counter, defaultdict

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DEFAULT_TEXT_DIR = "ncert_history_corpus_text"
DEFAULT_INDEX = "chapter_index_repaired.csv"
OUT_JSONL = "chunks.jsonl"
OUT_STATS = "chunk_stats.csv"

# Rough words->tokens factor. Avoids a tiktoken dependency; the embedding
# model's own tokenizer will differ slightly, so treat n_tokens_est as an
# estimate used for packing, not an exact count.
TOKENS_PER_WORD = 1.3

PAGE_RE = re.compile(r"^---\s*PAGE\s+(\d+)\s*---\s*$")

# Numbered in-chapter section headings, e.g. "1. Beginnings",
# "4.  The Royal Centre", "2.1 Something"
SECTION_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d)?)\.\s+([A-Z][^\n]{3,70})\s*$")

# Furniture patterns -> content_type
FURNITURE = [
    ("figure_caption", re.compile(r"^\s*Fig\.?\s*\d", re.I)),
    ("map_caption", re.compile(r"^\s*Map\s*\d", re.I)),
    ("source_box", re.compile(r"^\s*(Source|SOURCE)\s*\d*\s*$")),
    ("glossary", re.compile(r"^\s*(New words|KEYWORDS|Keywords)\s*$")),
    ("exercise", re.compile(r"(Let.s recall|Let.s discuss|Let.s do|"
                            r"ACTIVITY|Discuss\.\.\.|Answer in|"
                            r"Write a short essay)", re.I)),
    ("timeline", re.compile(r"^\s*\d{3,4}\s*[-–]\s*\d{2,4}\s*$")),
    ("reprint_stamp", re.compile(r"^\s*(Reprint\s+\d{4}-\d{2}|\d{4}-\d{2})\s*$")),
]


def decode_pua(text: str) -> str:
    """class8_our_pasts_3 encodes ~14k chars in the Private Use Area."""
    return "".join(
        chr(ord(c) - 0xF000) if 0xF000 <= ord(c) <= 0xF0FF else c for c in text
    )


def load_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    had_pua = any(0xF000 <= ord(c) <= 0xF0FF for c in raw)
    if had_pua:
        raw = decode_pua(raw)
    return raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"), had_pua


def classify_line(line: str) -> str:
    for label, pat in FURNITURE:
        if pat.search(line):
            return label
    return "prose"


def est_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


# ---------------------------------------------------------------------
# de-wrapping: the step that makes sentence packing possible
# ---------------------------------------------------------------------
SENT_END = re.compile(r"[.!?][\"')\]]?$")


def dewrap(lines):
    """
    Join hard-wrapped lines back into sentences.

    Yields dicts: {text, page, content_type}

    Rules:
      - "--- PAGE n ---" updates the current page and breaks the buffer
      - a line classified as furniture is emitted on its own (never glued to
        surrounding prose, or a caption ends up inside a paragraph)
      - otherwise lines accumulate until one ends with sentence punctuation
      - a word split across a line break by a trailing hyphen is rejoined
        ("impor-\\ntant" -> "important"), which is common in the OCR'd books
    """
    page = None
    buf = []
    buf_start_page = None

    def flush():
        nonlocal buf, buf_start_page
        if buf:
            text = " ".join(buf).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                yield {"text": text, "page": buf_start_page or page,
                       "content_type": "prose"}
            buf = []
            buf_start_page = None

    for raw in lines:
        line = raw.strip()

        m = PAGE_RE.match(line)
        if m:
            yield from flush()
            page = int(m.group(1))
            continue

        if not line:
            continue

        ctype = classify_line(line)
        if ctype != "prose":
            yield from flush()
            yield {"text": line, "page": page, "content_type": ctype}
            continue

        if buf_start_page is None:
            buf_start_page = page

        # rejoin a hyphen-split word
        if buf and buf[-1].endswith("-") and re.match(r"^[a-z]", line):
            buf[-1] = buf[-1][:-1] + line
        else:
            buf.append(line)

        if SENT_END.search(line):
            yield from flush()

    yield from flush()


# ---------------------------------------------------------------------
# packing units into chunks
# ---------------------------------------------------------------------
def pack(units, target_tokens, overlap_ratio, meta, start_index=0):
    """
    Pack a list of de-wrapped units into chunks of ~target_tokens, carrying
    overlap_ratio of the previous chunk's tail into the next chunk.

    Overlap is measured in tokens and applied at unit granularity, so a chunk
    never begins mid-sentence. This matters for the aggregation-style eval
    questions (H2, H9) whose answers straddle a boundary.
    """
    chunks = []
    cur, cur_tok = [], 0
    idx = start_index

    def emit(units_slice):
        nonlocal idx
        if not units_slice:
            return
        text = " ".join(u["text"] for u in units_slice).strip()
        if not text:
            return
        pages = [u["page"] for u in units_slice if u["page"] is not None]
        types = Counter(u["content_type"] for u in units_slice)
        # a chunk is "prose" if prose dominates, else named by its main type
        dominant = types.most_common(1)[0][0]
        mixed = len(types) > 1
        rec = dict(meta)
        rec.update({
            "chunk_id": f"{meta['book']}_c{idx:05d}",
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "content_type": dominant if not mixed else f"mixed:{dominant}",
            "n_tokens_est": est_tokens(text),
            "text": text,
        })
        chunks.append(rec)
        idx += 1

    for u in units:
        ut = est_tokens(u["text"])
        # A single oversized unit (a long timeline row, an OCR run-on) is
        # emitted alone rather than silently dropped or split mid-sentence.
        if ut >= target_tokens:
            emit(cur)
            cur, cur_tok = [], 0
            emit([u])
            continue

        if cur_tok + ut > target_tokens and cur:
            emit(cur)
            # build overlap tail from the end of the chunk just emitted
            tail, tail_tok = [], 0
            budget = target_tokens * overlap_ratio
            for prev in reversed(cur):
                pt = est_tokens(prev["text"])
                if tail_tok + pt > budget:
                    break
                tail.insert(0, prev)
                tail_tok += pt
            cur, cur_tok = list(tail), tail_tok

        cur.append(u)
        cur_tok += ut

    emit(cur)

    # Safety net: fold any runt chunk into its neighbour. Tiny chunks embed
    # poorly (too little context to be distinctive) and pollute recall@k by
    # occupying a slot without carrying an answer.
    MIN_TOKENS = 40
    merged = []
    for c in chunks:
        if (merged and c["n_tokens_est"] < MIN_TOKENS
                and merged[-1].get("section") == c.get("section")
                and merged[-1]["n_tokens_est"] + c["n_tokens_est"]
                    <= target_tokens * 1.4):
            prev = merged[-1]
            prev["text"] = (prev["text"] + " " + c["text"]).strip()
            prev["n_tokens_est"] = est_tokens(prev["text"])
            if c["page_end"] is not None:
                prev["page_end"] = max(
                    [p for p in (prev["page_end"], c["page_end"]) if p is not None])
            if c["page_start"] is not None and prev["page_start"] is None:
                prev["page_start"] = c["page_start"]
        else:
            merged.append(c)

    return merged, idx


def split_into_sections(units_with_lines):
    """
    Placeholder kept for clarity: section splitting is done on raw lines
    before de-wrapping, in chunk_chapter().
    """
    raise NotImplementedError


def chunk_chapter(lines, chap_meta, target_tokens, overlap, use_sections,
                  start_index):
    """
    Chunk one chapter's lines. If the chapter contains numbered section
    headings and use_sections is on, each section is packed independently so
    chunks never straddle a section boundary.
    """
    # locate section headings within this chapter
    sections = []
    for i, line in enumerate(lines):
        m = SECTION_RE.match(line.strip())
        if not m:
            continue
        title = m.group(2).strip()
        # A section heading is not a sentence. Numbered exercise items
        # ("1. Kings need money for armies, palaces, forts.") match the same
        # shape and were producing hundreds of 3-10 token chunks, so reject
        # anything that ends in terminal punctuation or reads as a question.
        if re.search(r"[.!?]$", title):
            continue
        if len(title.split()) > 8:
            continue
        # Reject prose fragments that happen to start with "N. Capital":
        #  - internal sentence punctuation ("...stone on stone. Here, the")
        #  - trailing lowercase function word, i.e. the line continues
        if re.search(r"[.!?]\s", title):
            continue
        if re.search(r"\b(the|a|an|of|and|in|to|for|with|that|this|is|was)$",
                     title, re.I):
            continue
        # Exercise prompts share the "N. Capital..." shape but are questions
        # or instructions, not headings. Without this, labels like
        # "1. Fill in the blanks:", "1. Write a one page history of Germany"
        # and "1. How is the trabeate principle..." become chunk boundaries
        # and fragment the exercise sections.
        if title.rstrip().endswith((":", "?")):
            continue
        if re.match(r"^(what|why|how|when|where|which|who|write|identify|"
                    r"fill|match|list|name|state|discuss|explain|describe|"
                    r"choose|complete|imagine|compare|give|find)\b",
                    title, re.I):
            continue
        sections.append((i, f"{m.group(1)}. {title}"))

    all_chunks = []
    idx = start_index

    if use_sections and len(sections) >= 2:
        bounds = [s[0] for s in sections] + [len(lines)]
        # anything before the first heading is chapter front-matter prose
        if bounds[0] > 0:
            meta = dict(chap_meta, section="")
            units = list(dewrap(lines[:bounds[0]]))
            c, idx = pack(units, target_tokens, overlap, meta, idx)
            all_chunks += c
        for k, (line_no, title) in enumerate(sections):
            seg = lines[bounds[k]:bounds[k + 1]]
            meta = dict(chap_meta, section=title)
            units = list(dewrap(seg))
            c, idx = pack(units, target_tokens, overlap, meta, idx)
            all_chunks += c
    else:
        meta = dict(chap_meta, section="")
        units = list(dewrap(lines))
        c, idx = pack(units, target_tokens, overlap, meta, idx)
        all_chunks += c

    return all_chunks, idx, len(sections)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dir", default=DEFAULT_TEXT_DIR)
    ap.add_argument("--index", default=DEFAULT_INDEX,
                    help="chapter index CSV (repaired one by default)")
    ap.add_argument("--target-tokens", type=int, default=400)
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--no-sections", action="store_true",
                    help="disable section-aware splitting (for A/B comparison)")
    ap.add_argument("--out", default=OUT_JSONL)
    args = ap.parse_args()

    if not os.path.isdir(args.text_dir):
        print(f"ERROR: {args.text_dir} not found.")
        return

    # gather text files
    paths = {}
    for dirpath, _, filenames in os.walk(args.text_dir):
        for fn in sorted(filenames):
            if fn.lower().endswith(".txt"):
                paths[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)

    # chapter index (optional -- files without an entry are chunked whole)
    chapters = defaultdict(list)
    if os.path.exists(args.index):
        for r in csv.DictReader(open(args.index, encoding="utf-8")):
            if r.get("start_line") in ("", None):
                continue
            chapters[r["book"]].append(r)
        for b in chapters:
            chapters[b].sort(key=lambda r: int(r["start_line"]))
        print(f"Loaded chapter index: {sum(len(v) for v in chapters.values())} "
              f"chapters across {len(chapters)} books")
    else:
        print(f"NOTE: {args.index} not found -- chunking without chapter "
              f"metadata. Run detect_chapters.py + repair_chapter_index.py "
              f"first for richer metadata.")

    all_chunks = []
    stats = []

    for book in sorted(paths):
        lines, had_pua = load_lines(paths[book])

        # source_type drives the clean-vs-OCR comparison in your eval
        if had_pua:
            source_type = "pua_decoded"
        elif book.startswith("old_ncert"):
            source_type = "ocr"
        else:
            source_type = "clean"

        # Chapter files from NCERT's ZIPs are already one chapter per file;
        # their chapter identity comes from the filename.
        is_chapter_file = book.startswith(("kehs", "lehs"))
        use_sections = not args.no_sections

        idx = 0
        book_chunks = []
        n_sections_total = 0

        if book in chapters and not is_chapter_file:
            for r in chapters[book]:
                s, e = int(r["start_line"]), int(r["end_line"])
                meta = {
                    "book": book,
                    "chapter_num": r.get("chapter_num", ""),
                    "chapter_title": r.get("chapter_title", ""),
                    "source_type": source_type,
                    "chapter_confidence": r.get("confidence", ""),
                }
                c, idx, nsec = chunk_chapter(
                    lines[s:e + 1], meta, args.target_tokens, args.overlap,
                    use_sections, idx)
                book_chunks += c
                n_sections_total += nsec
        else:
            meta = {
                "book": book,
                "chapter_num": "",
                "chapter_title": "",
                "source_type": source_type,
                "chapter_confidence": "n/a (one chapter per file)"
                                      if is_chapter_file else "",
            }
            c, idx, nsec = chunk_chapter(
                lines, meta, args.target_tokens, args.overlap,
                use_sections, idx)
            book_chunks += c
            n_sections_total += nsec

        all_chunks += book_chunks

        toks = [c["n_tokens_est"] for c in book_chunks] or [0]
        ctypes = Counter(c["content_type"] for c in book_chunks)
        stats.append({
            "book": book,
            "source_type": source_type,
            "n_chunks": len(book_chunks),
            "n_sections_found": n_sections_total,
            "median_tokens": sorted(toks)[len(toks) // 2],
            "min_tokens": min(toks),
            "max_tokens": max(toks),
            "pct_prose": round(
                100 * sum(v for k, v in ctypes.items()
                          if k.startswith("prose") or k == "mixed:prose")
                / max(1, len(book_chunks)), 1),
        })
        print(f"  {book[:42]:44s} {len(book_chunks):5d} chunks  "
              f"[{source_type:12s}] median {stats[-1]['median_tokens']:4d} tok, "
              f"{stats[-1]['pct_prose']:5.1f}% prose, "
              f"{n_sections_total} sections")

    # write chunks
    with open(args.out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(OUT_STATS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)

    # summary
    toks = [c["n_tokens_est"] for c in all_chunks]
    ctypes = Counter(c["content_type"] for c in all_chunks)
    srcs = Counter(c["source_type"] for c in all_chunks)
    print(f"\n{len(all_chunks)} chunks -> {args.out}")
    print(f"  token estimate: median {sorted(toks)[len(toks)//2]}, "
          f"min {min(toks)}, max {max(toks)}")
    print(f"  by source_type: {dict(srcs)}")
    print(f"  top content types: "
          f"{dict(ctypes.most_common(6))}")
    print(f"  stats -> {OUT_STATS}")
    print("\nNext: embed these with sentence-transformers and load into "
          "Chroma. Keep source_type in the metadata -- it is what lets you "
          "report clean-vs-OCR retrieval quality separately.")


if __name__ == "__main__":
    main()