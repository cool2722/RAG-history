#!/usr/bin/env python3
"""
Extract text from every PDF in the NCERT corpus for RAG ingestion.

Handles two kinds of PDFs automatically:
  - "real text" PDFs (most official NCERT files)      -> fast, direct extraction
  - scanned/image PDFs (e.g. the Old NCERT reprints)   -> OCR fallback (tesseract)

For each source PDF at:
    ncert_history_corpus/<book>/<file>.pdf
this writes a plain-text file at:
    ncert_history_corpus_text/<book>/<file>.txt
plus one page-marker per page (--- PAGE n ---) so you can keep
page-level provenance for citations later.

It also writes ncert_history_corpus_text/extraction_report.csv
recording, per file: page count, method used (text/ocr/mixed),
and average characters extracted per page (a quick quality signal
-- a very low number on a "text" file usually means it's actually
scanned and slipped through, worth checking by hand).

DUPLICATE HANDLING
------------------
Two safeguards against duplicate content in your corpus:
  1. If a PDF's output .txt already exists, it's skipped entirely.
     Safe to re-run after a crash/interruption -- won't redo work,
     and won't re-run expensive OCR unnecessarily.
  2. If a PDF's content (by hash) is identical to one already
     processed elsewhere in the corpus (e.g. the same chapter
     sitting in two differently-named folders), the duplicate is
     *not* re-OCR'd -- its text is copied from the original instead.
     This also means duplicate text never ends up in your corpus
     twice under two different paths.

SETUP
-----
pip install pymupdf pytesseract pillow

OCR also needs the Tesseract binary itself (not just the Python
wrapper):
  - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
             then either add it to PATH, or set TESSERACT_CMD below
             to the full path of tesseract.exe
  - macOS:   brew install tesseract
  - Linux:   sudo apt install tesseract-ocr
"""

import os
import csv
import time
import shutil
import hashlib
import fitz  # pymupdf

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
INPUT_DIR = "ncert_history_corpus"
OUTPUT_DIR = "ncert_history_corpus_text"

# If a page's direct text extraction yields fewer than this many
# characters, treat the page as "probably scanned" and OCR it instead.
MIN_CHARS_FOR_TEXT_PAGE = 20

# OCR render resolution -- 300 is a good quality/speed tradeoff.
OCR_DPI = 300

# On Windows, if tesseract isn't on PATH, uncomment and set the path:
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_pdf(pdf_path: str):
    """
    Returns (full_text, method, pages_ocred, total_pages)
    method is 'text', 'ocr', or 'mixed'.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    page_texts = []
    pages_ocred = 0
    ocr_available = None  # lazy-import check, only if needed

    for page_num, page in enumerate(doc):
        if page_num > 0 and page_num % 20 == 0:
            print(f"    ...page {page_num}/{total_pages}")
        text = page.get_text().strip()

        if len(text) >= MIN_CHARS_FOR_TEXT_PAGE:
            page_texts.append(text)
            continue

        # Fall back to OCR for this page
        if ocr_available is None:
            try:
                import pytesseract
                from PIL import Image
                ocr_available = True
            except ImportError:
                ocr_available = False
                print("    WARNING: pytesseract/Pillow not installed, "
                      "skipping OCR for low-text pages.")

        if not ocr_available:
            page_texts.append(text)  # keep whatever little text there was
            continue

        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            ocr_text = pytesseract.image_to_string(img).strip()
        except Exception as e:
            print(f"    OCR failed on page {page_num}: {e}")
            ocr_text = text
        page_texts.append(ocr_text)
        pages_ocred += 1

    doc.close()

    if pages_ocred == 0:
        method = "text"
    elif pages_ocred == total_pages:
        method = "ocr"
    else:
        method = "mixed"

    full_text = "\n\n".join(
        f"--- PAGE {i + 1} ---\n{t}" for i, t in enumerate(page_texts)
    )
    return full_text, method, pages_ocred, total_pages


def file_hash(path: str) -> str:
    """MD5 of file content, in chunks so large PDFs don't blow up memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"ERROR: '{INPUT_DIR}' not found. Run this from the folder "
              f"that contains it, or edit INPUT_DIR at the top of the script.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_rows = []

    # Load any prior report so that skipped (already-done) files still
    # show up correctly in the new report instead of vanishing.
    report_path = os.path.join(OUTPUT_DIR, "extraction_report.csv")
    prior_report = {}
    if os.path.exists(report_path):
        with open(report_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prior_report[(row["book"], row["file"])] = row

    # hash -> (source pdf path, already-extracted .txt path)
    # Pre-populated from any run of this script that already completed,
    # and grown as we go -- catches duplicates both from before this run
    # and duplicates discovered within this same run.
    seen_hashes = {}

    for book in sorted(os.listdir(INPUT_DIR)):
        book_in = os.path.join(INPUT_DIR, book)
        if not os.path.isdir(book_in):
            continue  # skip manifest.csv etc.

        book_out = os.path.join(OUTPUT_DIR, book)
        os.makedirs(book_out, exist_ok=True)

        for fname in sorted(os.listdir(book_in)):
            if not fname.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(book_in, fname)
            out_name = os.path.splitext(fname)[0] + ".txt"
            out_path = os.path.join(book_out, out_name)

            # Skip 1: already extracted in a previous run of this script.
            if os.path.exists(out_path):
                print(f"Skipping {book}/{fname}: already extracted -> {out_path}")
                prior = prior_report.get((book, fname))
                if prior:
                    report_rows.append(prior)
                else:
                    # .txt exists but wasn't in a prior report (e.g. report
                    # was deleted) -- record what we can without redoing work.
                    report_rows.append({
                        "book": book, "file": fname, "pages": "unknown",
                        "method": "already-done", "pages_ocred": "unknown",
                        "avg_chars_per_page": "unknown",
                    })
                # Register its hash so later duplicates still resolve
                # correctly, without re-reading the whole PDF unnecessarily.
                try:
                    seen_hashes[file_hash(pdf_path)] = (pdf_path, out_path)
                except OSError:
                    pass
                continue

            print(f"Processing {book}/{fname} ...")
            h = file_hash(pdf_path)

            # Skip 2: identical content already processed (this run or
            # a prior one), e.g. the same chapter sitting in two
            # differently-named folders. Reuse its text instead of
            # redoing (possibly expensive) OCR.
            if h in seen_hashes:
                src_pdf, src_txt = seen_hashes[h]
                print(f"  Duplicate of {src_pdf} -- copying its extracted "
                      f"text instead of re-processing.")
                shutil.copyfile(src_txt, out_path)
                report_rows.append({
                    "book": book, "file": fname, "pages": "duplicate",
                    "method": "duplicate", "pages_ocred": 0,
                    "avg_chars_per_page": "see " + src_pdf,
                })
                continue

            start = time.time()
            try:
                text, method, pages_ocred, total_pages = extract_pdf(pdf_path)
            except Exception as e:
                print(f"  FAILED: {e}")
                report_rows.append({
                    "book": book, "file": fname, "pages": "ERROR",
                    "method": "error", "pages_ocred": 0,
                    "avg_chars_per_page": 0,
                })
                continue

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            seen_hashes[h] = (pdf_path, out_path)

            avg_chars = len(text) / max(total_pages, 1)
            elapsed = time.time() - start
            print(f"  -> {out_path}  [{method}, {pages_ocred}/{total_pages} "
                  f"pages OCR'd, ~{avg_chars:.0f} chars/page, {elapsed:.1f}s]")

            report_rows.append({
                "book": book, "file": fname, "pages": total_pages,
                "method": method, "pages_ocred": pages_ocred,
                "avg_chars_per_page": round(avg_chars, 1),
            })

    report_path = os.path.join(OUTPUT_DIR, "extraction_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["book", "file", "pages", "method",
                           "pages_ocred", "avg_chars_per_page"]
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nDone. Report written to {report_path}")
    print("Check rows with low avg_chars_per_page on 'text' method files -- "
          "that usually means a PDF is scanned but slipped past the OCR check.")


if __name__ == "__main__":
    main()