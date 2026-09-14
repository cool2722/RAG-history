#!/usr/bin/env python3
"""
Repair chapter_index.csv without checking 54 rows by hand.

THE PROBLEM
-----------
detect_chapters.py flags rows "low" confidence. Inspecting the 54 low rows
shows they are not 54 independent problems but five recurring patterns:

  1. TOC artifacts        11 rows -- "chapters" spanning 2-135 lines, all
                                     sitting on printed pages 7-8. These are
                                     table-of-contents entries, not chapters.
  2. Running-header titles 6 rows -- title came out as the BOOK header
                                     ("India and the Contemporary World").
  3. Garbage/OCR titles    6 rows -- '¥', '‘', '- The Later Vedic Phase',
                                     '-~"* Economic Impact of the British Rule'.
  4. Duplicate numbers    ~13 rows-- modern_india has ch7 three times.
  5. Numbering gaps        9      -- genuinely undetected chapters.

THE FIX
-------
Each book contains its own table of contents. We use it as ground truth via
MONOTONIC SEQUENCE ALIGNMENT:

    detected chapters (in document order)  <->  TOC entries (in TOC order)

Alignment is scored by fuzzy title similarity but constrained to preserve
order. This is what makes it robust to the three worst cases:
  - a garbage title ('¥') still gets the right TOC entry from its POSITION
  - a missing chapter shows up as an unmatched TOC entry (a real gap)
  - a duplicate detection shows up as two detections competing for one TOC
    entry; only the better-scoring one wins

Why not page-number alignment? It was verified to work perfectly for class9
(TOC printed pages 3/25/49/77/97/117 map to PDF pages 15/61/89/109/129 at a
constant offset of 12). But in ancient_india and medieval_india the OCR
decoupled TOC titles from their page numbers (titles on lines 120-145, the
numbers dumped together on lines 147-160), so page alignment is not generally
available. Where page numbers ARE adjacent to titles, we use them as an
independent cross-check and report whether they agree.

OUTPUT
------
  chapter_index_repaired.csv   repaired rows + provenance of every change
  repair_report.txt            human-readable summary, including an explicit
                               list of what could NOT be fixed

Nothing is silently "fixed". Every changed field records why, and anything the
script is unsure about is left flagged rather than guessed.

USAGE
-----
  python repair_chapter_index.py
  python repair_chapter_index.py --index chapter_index.csv --text-dir DIR
"""

import os
import re
import csv
import sys
import argparse
from difflib import SequenceMatcher

MIN_CHAPTER_SPAN = 200      # lines; below this it's a TOC artifact, not a chapter
FRONT_MATTER_LIMIT = 400    # lines to scan for a table of contents
TITLE_MATCH_FLOOR = 0.45    # below this, a title match is not trusted


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def norm_title(s: str) -> str:
    """Aggressive normalisation so OCR damage doesn't block a match."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similar(a: str, b: str) -> float:
    a, b = norm_title(a), norm_title(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def is_junk_title(t: str) -> bool:
    t = (t or "").strip()
    if len(t) < 4:
        return True
    alpha = sum(c.isalpha() or c.isspace() for c in t) / max(1, len(t))
    return alpha < 0.75 or bool(re.match(r'^[-_~"\'`¥\s]', t))


def load_lines(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    # match detect_chapters.py: decode Private Use Area (class8)
    if any(0xF000 <= ord(c) <= 0xF0FF for c in raw):
        raw = "".join(
            chr(ord(c) - 0xF000) if 0xF000 <= ord(c) <= 0xF0FF else c for c in raw
        )
    return raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")


# ---------------------------------------------------------------------
# TOC extraction -- several real formats, all seen in this corpus
# ---------------------------------------------------------------------
ROMAN_WORD = r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI|XXII|XXIII|XXIV|XXV|XXVI|XXVII)"

# "I. The French Revolution"  /  "7. Weavers, Iron Smelters"
TOC_INLINE = re.compile(
    rf"^\s*({ROMAN_WORD}|\d{{1,2}})\s*[.)]\s+(.{{6,90}}?)\s*$", re.I
)
# "CHAPTER I" (or OCR-mangled "CHAPTIR V") on its own line, title follows
TOC_MARKER = re.compile(
    rf"^\s*(?:CHAPTER|CHAPTIR|CHAPFER|CHAPTHR)\s+({ROMAN_WORD}|\d{{1,2}})\s*\.?\s*$",
    re.I,
)


def extract_toc(lines):
    """
    Return an ordered list of {'title':..., 'page':int|None, 'line':int}.
    Order is TOC order; that ordering is what the alignment relies on.

    The window is bounded to the CONTIGUOUS contents block. Without this,
    scanning a fixed 400 lines runs past the TOC and swallows the real
    chapter-1 body as an extra "TOC entry" (seen on modern_india, where the
    body heading at line 180 was being treated as a 14th contents row).
    """
    window = lines[:FRONT_MATTER_LIMIT]
    entries = []

    def bound_to_block(found):
        """
        Keep the LONGEST contiguous run of entries.

        Not the first run: on class9 the first run begins at a stray
        constitution-preamble line ("Subs. by the Constitution (Forty-second
        Amendment)") in the front matter, and taking it discarded the real
        contents block a few lines later. The genuine TOC is the longest
        tightly-spaced run.
        """
        if len(found) < 2:
            return found
        runs, cur = [], [found[0]]
        for prev, nxt in zip(found, found[1:]):
            if nxt["line"] - prev["line"] > 12:
                runs.append(cur)
                cur = [nxt]
            else:
                cur.append(nxt)
        runs.append(cur)
        return max(runs, key=len)

    # --- format A: number/roman + title on one line ---
    for i, line in enumerate(window):
        m = TOC_INLINE.match(line)
        if not m:
            continue
        title = m.group(2).strip(" .-—_")
        if len(norm_title(title)) < 6:
            continue
        page = None
        for j in range(i + 1, min(i + 4, len(window))):
            s = window[j].strip()
            if re.fullmatch(r"\d{1,3}", s):
                page = int(s)
                break
        entries.append({"title": title, "page": page, "line": i})
    entries = bound_to_block(entries)

    # --- format B: "CHAPTER N" marker, title on a following line ---
    if len(entries) < 3:
        entries = []
        for i, line in enumerate(window):
            if not TOC_MARKER.match(line):
                continue
            title, page = None, None
            for j in range(i + 1, min(i + 4, len(window))):
                s = window[j].strip()
                if not s:
                    continue
                if re.fullmatch(r"\d{1,3}", s):
                    page = page or int(s)
                    continue
                if len(norm_title(s)) >= 6 and title is None:
                    title = s.strip(" .-—_")
            if title:
                entries.append({"title": title, "page": page, "line": i})
        entries = bound_to_block(entries)

    # --- format C deliberately omitted ---
    # An earlier version had a loose "grab title-ish lines after 'Contents'"
    # fallback. On class8 it scraped the copyright page ("ALL RIGHTS RESERVED",
    # "Phone : 011-26562708", "PD 460T MJ") and treated them as chapters.
    # A TOC parser that invents entries is worse than one that admits defeat,
    # so if neither format A nor B matches we fall back to order-only repair.

    # de-duplicate consecutive near-identical titles, keep TOC order
    cleaned = []
    for e in entries:
        if cleaned and similar(e["title"], cleaned[-1]["title"]) > 0.9:
            continue
        cleaned.append(e)
    return cleaned


def recover_title_locally(lines, start_line, window=12):
    """
    For a junk title, look near the chapter start for a better heading
    candidate. Used when the book's TOC is unusable, so there is no
    authoritative title list to match against.
    """
    best, best_score = None, 0.0
    lo = max(0, start_line - 2)
    hi = min(len(lines), start_line + window)
    for i in range(lo, hi):
        s = lines[i].strip()
        if not (8 <= len(s) <= 70):
            continue
        if is_junk_title(s):
            continue
        if re.match(r"---\s*PAGE", s):
            continue
        # Skip the chapter marker itself: "CHAPTER 8" is not a title.
        if re.match(r"^\s*(CHAPTER|CHAPTIR|Chapter)\s+([IVXLC]+|\d{1,2})\s*\.?\s*$", s):
            continue
        # prefer heading-like lines: title case or caps, few digits
        score = 0.0
        if s.istitle() or s == s.upper():
            score += 0.5
        digits = sum(c.isdigit() for c in s) / max(1, len(s))
        score += max(0.0, 0.5 - digits * 2)
        if score > best_score:
            best, best_score = s, score
    return best if best_score >= 0.45 else None


# ---------------------------------------------------------------------
# monotonic alignment: detections <-> TOC entries
# ---------------------------------------------------------------------
def align(detections, toc, gap_penalty=0.25):
    """
    Needleman-Wunsch style alignment preserving order.
    detections: list of dicts with 'chapter_title'
    toc:        list of dicts with 'title'
    Returns list of (det_idx|None, toc_idx|None, score).
    """
    n, m = len(detections), len(toc)
    if n == 0 or m == 0:
        return [(i, None, 0.0) for i in range(n)]

    # score matrix
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        S[i][0] = S[i - 1][0] - gap_penalty
    for j in range(1, m + 1):
        S[0][j] = S[0][j - 1] - gap_penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dt = detections[i - 1].get("chapter_title", "")
            tt = toc[j - 1]["title"]
            # A junk detected title carries no evidence either way, so give it
            # a neutral score and let position (monotonicity) decide.
            sim = 0.35 if is_junk_title(dt) else similar(dt, tt)
            S[i][j] = max(
                S[i - 1][j - 1] + sim,
                S[i - 1][j] - gap_penalty,
                S[i][j - 1] - gap_penalty,
            )

    # traceback
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        dt = detections[i - 1].get("chapter_title", "")
        tt = toc[j - 1]["title"]
        sim = 0.35 if is_junk_title(dt) else similar(dt, tt)
        if abs(S[i][j] - (S[i - 1][j - 1] + sim)) < 1e-9:
            pairs.append((i - 1, j - 1, sim))
            i, j = i - 1, j - 1
        elif abs(S[i][j] - (S[i - 1][j] - gap_penalty)) < 1e-9:
            pairs.append((i - 1, None, 0.0))
            i -= 1
        else:
            pairs.append((None, j - 1, 0.0))
            j -= 1
    while i > 0:
        pairs.append((i - 1, None, 0.0))
        i -= 1
    while j > 0:
        pairs.append((None, j - 1, 0.0))
        j -= 1

    pairs.reverse()
    return pairs


def check_page_offset(matched):
    """
    Independent cross-check: where TOC page numbers survived, do detected PDF
    pages differ from printed pages by a CONSTANT offset?
    Returns (offset, n_agree, n_total) or (None, 0, 0).
    """
    offsets = []
    for det, toc_e, _ in matched:
        if det is None or toc_e is None:
            continue
        if toc_e.get("page") is None or not det.get("start_page"):
            continue
        try:
            offsets.append(int(det["start_page"]) - int(toc_e["page"]))
        except (TypeError, ValueError):
            continue
    if len(offsets) < 3:
        return None, 0, len(offsets)
    from collections import Counter
    off, n = Counter(offsets).most_common(1)[0]
    return off, n, len(offsets)


# ---------------------------------------------------------------------
# targeted search for chapters the detector missed entirely
# ---------------------------------------------------------------------
def find_missing_chapter(lines, title, lo, hi):
    """
    Search a bounded line range for a known chapter title.
    Far more reliable than the generic detector because we know both the
    expected string and roughly where it should be.
    Returns line index or None.
    """
    target = norm_title(title)
    if len(target) < 8:
        return None
    lo = max(0, lo)
    hi = min(len(lines), hi)
    best, best_score = None, 0.0
    for i in range(lo, hi):
        s = lines[i].strip()
        if not (6 <= len(s) <= 90):
            continue
        sc = similar(s, title)
        # prefer heading-like lines (short, caps or title case)
        if s == s.upper() or s.istitle():
            sc += 0.08
        if sc > best_score:
            best, best_score = i, sc
    return best if best_score >= 0.62 else None


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="chapter_index.csv")
    ap.add_argument("--text-dir", default="ncert_history_corpus_text")
    ap.add_argument("--out", default="chapter_index_repaired.csv")
    ap.add_argument("--report", default="repair_report.txt")
    args = ap.parse_args()

    if not os.path.exists(args.index):
        print(f"ERROR: {args.index} not found.")
        sys.exit(1)

    rows = list(csv.DictReader(open(args.index, encoding="utf-8")))

    # locate the .txt for each book
    text_paths = {}
    for dirpath, _, filenames in os.walk(args.text_dir):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                text_paths[os.path.splitext(fn)[0]] = os.path.join(dirpath, fn)

    out_rows = []
    report = []
    unfixed = []

    books = []
    for r in rows:
        if r["book"] not in books:
            books.append(r["book"])

    for book in books:
        brows = [r for r in rows if r["book"] == book and r.get("start_line") != ""]
        if not brows:
            continue

        report.append(f"\n=== {book} ===")

        # ---- Step 1: drop TOC artifacts by span ----
        kept, dropped = [], []
        for r in brows:
            try:
                span = int(r["end_line"]) - int(r["start_line"])
            except ValueError:
                dropped.append((r, "unparseable line numbers"))
                continue
            if span < MIN_CHAPTER_SPAN:
                dropped.append((r, f"span {span} < {MIN_CHAPTER_SPAN}"))
            else:
                kept.append(r)

        if dropped:
            report.append(f"  [1] dropped {len(dropped)} TOC artifact(s):")
            for r, why in dropped[:12]:
                report.append(f"        ch{r['chapter_num']:>3} "
                              f"'{r['chapter_title'][:38]}' ({why})")

        if not kept:
            report.append("  ! every row dropped -- nothing to repair")
            unfixed.append(f"{book}: all rows looked like TOC artifacts")
            continue

        kept.sort(key=lambda r: int(r["start_line"]))

        # Only books with at least one LOW row need TOC repair. Touching a
        # book that is already consistent risks making it worse (class8 is
        # 'medium' via title_bands and is internally coherent).
        if not any(r["confidence"] == "low" for r in kept):
            report.append(f"  [ok] {len(kept)} chapters, no low-confidence "
                          f"rows -- left untouched")
            for r in kept:
                r.setdefault("repair_notes", "")
                out_rows.append(r)
            continue

        # ---- Step 2: extract the book's own TOC ----
        path = text_paths.get(book)
        if not path:
            report.append(f"  ! text file for {book} not found under {args.text_dir}")
            unfixed.append(f"{book}: source .txt not found, left as-is")
            for r in kept:
                r["repair_notes"] = "source text missing; not repaired"
                out_rows.append(r)
            continue

        lines = load_lines(path)
        toc = extract_toc(lines)
        report.append(f"  [2] TOC entries found: {len(toc)}")
        if toc:
            for e in toc[:4]:
                report.append(f"        '{e['title'][:50]}' (page {e['page']})")

        if len(toc) < 4:
            # ---- order-only repair (no usable TOC) ----
            # ancient_india and medieval_india land here: OCR decoupled the
            # TOC titles from their page numbers, so there is no authoritative
            # list to align against. Their DETECTED titles are mostly fine
            # though ("The Age of the Mauryas", "The Geographical Setting"),
            # so the real problems are duplicate/non-sequential numbering and
            # a handful of junk titles. Fix those without inventing anything.
            report.append(f"  [2] TOC unusable ({len(toc)} entries) "
                          f"-> order-only repair")
            fixed_titles = 0
            still_junk = []
            for r in kept:
                r.setdefault("repair_notes", "")
                if is_junk_title(r["chapter_title"]):
                    better = recover_title_locally(lines, int(r["start_line"]))
                    if better:
                        old = r["chapter_title"]
                        r["chapter_title"] = better
                        r["repair_notes"] = (
                            f"junk title {old[:18]!r} replaced from nearby "
                            f"heading (no TOC available)")
                        fixed_titles += 1
                    else:
                        still_junk.append(r)

            kept.sort(key=lambda r: int(r["start_line"]))
            for k, r in enumerate(kept):
                old_num = r["chapter_num"]
                r["chapter_num"] = k + 1
                if str(old_num) != str(k + 1):
                    r["repair_notes"] = (
                        r["repair_notes"] + "; " if r["repair_notes"] else ""
                    ) + f"renumbered {old_num}->{k+1} (document order)"
                r["end_line"] = (int(kept[k + 1]["start_line"]) - 1
                                 if k + 1 < len(kept) else len(lines) - 1)
                # No independent corroboration available, so do not claim high.
                r["confidence"] = "medium"
                r["repair_notes"] = (r["repair_notes"] +
                                     "; order-only (unverified titles)").strip("; ")

            report.append(f"  [5] recovered {fixed_titles} junk title(s); "
                          f"{len(still_junk)} still unusable")
            for r in still_junk:
                unfixed.append(f"{book}: ch{r['chapter_num']} title still junk "
                               f"({r['chapter_title'][:20]!r}) @line {r['start_line']}")
            report.append(f"  => {len(kept)} chapters after order-only repair")
            out_rows.extend(kept)
            continue

        # ---- Step 3: monotonic alignment ----
        pairs = align(kept, toc)
        matched = []
        for di, ti, sc in pairs:
            det = kept[di] if di is not None else None
            toc_e = toc[ti] if ti is not None else None
            matched.append((det, toc_e, sc))

        n_pairs = sum(1 for d, t, _ in matched if d and t)
        report.append(f"  [3] aligned {n_pairs} detection(s) to TOC entries")

        # ---- Step 4: page-offset cross-check ----
        off, agree, total = check_page_offset(matched)
        if off is not None:
            report.append(f"  [4] page-offset cross-check: offset={off}, "
                          f"{agree}/{total} chapters agree")
            if agree < total * 0.6:
                report.append("        (weak agreement -- treating alignment "
                              "as unverified)")
        else:
            report.append("  [4] page-offset cross-check unavailable "
                          "(TOC page numbers missing or OCR-decoupled)")

        # ---- Step 5: apply repairs ----
        repaired = []
        for idx, (det, toc_e, sc) in enumerate(matched):
            if det is None and toc_e is None:
                continue

            if det is None:
                # TOC entry with no detection = a genuinely missing chapter.
                # Bound the search by its neighbours and go looking for it.
                lo, hi = 0, len(lines)
                prev = next((d for d, t, _ in matched[:idx][::-1] if d), None)
                nxt = next((d for d, t, _ in matched[idx + 1:] if d), None)
                if prev:
                    lo = int(prev["start_line"]) + 1
                if nxt:
                    hi = int(nxt["start_line"])
                found = find_missing_chapter(lines, toc_e["title"], lo, hi)
                # A recovered boundary must leave a plausible chapter behind it.
                # Without this, class9's recovered "The French Revolution"
                # landed 17 lines before the next chapter -- almost certainly a
                # contents line, not a chapter start.
                if found is not None and nxt is not None:
                    if int(nxt["start_line"]) - found < MIN_CHAPTER_SPAN:
                        report.append(
                            f"  [5] rejected recovery of "
                            f"'{toc_e['title'][:38]}' @line {found}: only "
                            f"{int(nxt['start_line']) - found} lines before "
                            f"next chapter")
                        unfixed.append(
                            f"{book}: chapter '{toc_e['title'][:40]}' "
                            f"recovery rejected (implausible span)")
                        found = None
                if found is not None:
                    repaired.append({
                        "book": book,
                        "chapter_num": "",  # assigned after ordering
                        "chapter_title": toc_e["title"],
                        "start_line": found,
                        "end_line": "",
                        "start_page": "",
                        "detection_method": "toc_gap_search",
                        "confidence": "medium",
                        "pua_decoded": kept[0].get("pua_decoded", ""),
                        "repair_notes": f"recovered missing chapter in lines "
                                        f"{lo}-{hi} via TOC title search",
                    })
                    report.append(f"  [5] recovered missing chapter "
                                  f"'{toc_e['title'][:42]}' @line {found}")
                else:
                    report.append(f"  [5] MISSING, not recoverable: "
                                  f"'{toc_e['title'][:42]}' "
                                  f"(searched lines {lo}-{hi})")
                    unfixed.append(f"{book}: chapter '{toc_e['title'][:40]}' "
                                   f"not found in lines {lo}-{hi}")
                continue

            if toc_e is None:
                # detection with no TOC entry -- likely a spurious split
                det["repair_notes"] = ("no matching TOC entry; possible "
                                       "spurious boundary -- review")
                det["confidence"] = "low"
                repaired.append(det)
                unfixed.append(f"{book}: detection ch{det['chapter_num']} "
                               f"'{det['chapter_title'][:30]}' has no TOC match")
                continue

            notes = []
            old_title = det["chapter_title"]
            if is_junk_title(old_title) or sc < TITLE_MATCH_FLOOR:
                det["chapter_title"] = toc_e["title"]
                notes.append(f"title replaced from TOC (was {old_title[:24]!r})")
            elif sc < 0.95 and norm_title(old_title) != norm_title(toc_e["title"]):
                det["chapter_title"] = toc_e["title"]
                notes.append(f"title normalised to TOC (sim={sc:.2f})")

            det["_toc_order"] = toc.index(toc_e)
            det["repair_notes"] = "; ".join(notes)
            repaired.append(det)

        # ---- Step 6: renumber by document order, recompute end_line ----
        repaired.sort(key=lambda r: int(r["start_line"]))
        for k, r in enumerate(repaired):
            old_num = r.get("chapter_num", "")
            r["chapter_num"] = k + 1
            if str(old_num) != str(k + 1) and old_num != "":
                extra = f"renumbered {old_num}->{k+1} (document order)"
                r["repair_notes"] = (r.get("repair_notes") or "")
                r["repair_notes"] = (r["repair_notes"] + "; " + extra).strip("; ")

        for k, r in enumerate(repaired):
            r["end_line"] = (int(repaired[k + 1]["start_line"]) - 1
                             if k + 1 < len(repaired) else len(lines) - 1)
            if not r.get("start_page"):
                for j in range(int(r["start_line"]), -1, -1):
                    m = re.match(r"---\s*PAGE\s+(\d+)\s*---", lines[j].strip())
                    if m:
                        r["start_page"] = int(m.group(1))
                        break
            # confidence: upgrade only if alignment was corroborated
            if r["detection_method"] != "toc_gap_search":
                if off is not None and agree >= total * 0.6:
                    r["confidence"] = "high"
                    r["repair_notes"] = (r.get("repair_notes", "") +
                                         "; TOC+page-offset corroborated").strip("; ")
                else:
                    r["confidence"] = "medium"
                    r["repair_notes"] = (r.get("repair_notes", "") +
                                         "; TOC-aligned (title only)").strip("; ")
            r.pop("_toc_order", None)

        report.append(f"  => {len(repaired)} chapters after repair "
                      f"(was {len(brows)} rows)")
        out_rows.extend(repaired)

    # ---- write output ----
    fields = ["book", "chapter_num", "chapter_title", "start_line", "end_line",
              "start_page", "detection_method", "confidence", "pua_decoded",
              "repair_notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            r.setdefault("repair_notes", "")
            w.writerow(r)

    # ---- summary ----
    from collections import Counter
    conf = Counter(r["confidence"] for r in out_rows)
    summary = [
        "CHAPTER INDEX REPAIR REPORT",
        "=" * 60,
        f"input rows      : {len(rows)}",
        f"output chapters : {len(out_rows)}",
        f"confidence      : " + ", ".join(f"{k}={v}" for k, v in conf.most_common()),
        "",
        "NOT FIXED (needs a human look):" if unfixed else "NOT FIXED: none",
    ]
    summary += [f"  - {u}" for u in unfixed]
    summary += ["", "PER-BOOK DETAIL"] + report

    text = "\n".join(summary)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(text)
    print(f"\nWrote {args.out} and {args.report}")
    print("\nNext: spot-check a stratified sample (~10 rows across detection "
          "methods) rather than all of them, and record the observed error "
          "rate. That is a claim a reader can actually verify.")


if __name__ == "__main__":
    main()