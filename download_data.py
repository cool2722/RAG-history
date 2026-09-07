#!/usr/bin/env python3
"""
Download NCERT History textbooks for a RAG corpus.

Source of links: Vajiram & Ravi's NCERT-for-UPSC page, which hosts
whole-book PDFs directly (classes 6-10, on their own CDN) or links
straight to NCERT's own ZIPs (classes 11-12, each ZIP = one PDF per
chapter). See:
https://vajiramandravi.com/upsc-exam/ncert-books-for-upsc/

Two kinds of entries in BOOKS below:
  - "pdf": a single URL that is already a whole-book PDF -> saved as-is.
  - "zip": a URL to an NCERT ZIP -> downloaded and unzipped into
           one PDF per chapter.

This script:
  1. Scans the whole corpus folder first and hashes every existing PDF,
     so it knows what you already have -- including files sitting in
     oddly-named folders (like a manually extracted "lehs1dd").
  2. Downloads every book listed in BOOKS, skipping any file whose
     content (by hash) already exists anywhere in the corpus.
  3. Skips re-downloading anything already saved at its expected path
     (safe to re-run after an interruption).
  4. Unzips the ZIP entries into per-chapter PDFs, skipping chapters
     that duplicate a file already present elsewhere.
  5. Writes a manifest.csv (book, class, file, source_url) for your
     ingestion pipeline to use as metadata / citation info.
"""

import os
import time
import hashlib
import zipfile
import requests

# ---------------------------------------------------------------------
# CONFIG: History books only. label -> (type, url, class_level)
# ---------------------------------------------------------------------
BOOKS = {
    "class6_our_pasts_1": (
        "pdf",
        "https://vajiramias.sgp1.cdn.digitaloceanspaces.com/wp/upsc-exam/2025/10/ncert-class-6-social-science-book-pdf-our-pasts-1.pdf",
        "6",
    ),
    "class7_our_pasts_2": (
        "pdf",
        "https://vajiramias.sgp1.cdn.digitaloceanspaces.com/wp/upsc-exam/2025/10/ncert-class-7-social-science-book-pdf-history-our-pasts-ii.pdf",
        "7",
    ),
    "class8_our_pasts_3": (
        "pdf",
        "https://vajiramias.sgp1.cdn.digitaloceanspaces.com/wp/upsc-exam/2025/10/ncert-class-8-social-science-book-pdf-history-our-pasts-III.pdf",
        "8",
    ),
    "class9_india_and_contemporary_world_1": (
        "pdf",
        "https://vajiramias.sgp1.cdn.digitaloceanspaces.com/wp/upsc-exam/2025/10/ncert-class-9-social-science-book-pdf-history.pdf",
        "9",
    ),
    "class10_india_and_contemporary_world_2": (
        "pdf",
        "https://vajiramias.sgp1.cdn.digitaloceanspaces.com/wp/upsc-exam/2025/10/ncert-class-10-social-science-book-pdf-history.pdf",
        "10",
    ),
    "class11_themes_in_world_history": (
        "zip",
        "https://ncert.nic.in/textbook/pdf/kehs1dd.zip",
        "11",
    ),
    "class12_themes_in_indian_history_1": (
        "zip",
        "https://ncert.nic.in/textbook/pdf/lehs1dd.zip",
        "12",
    ),
    "class12_themes_in_indian_history_2": (
        "zip",
        "https://ncert.nic.in/textbook/pdf/lehs2dd.zip",
        "12",
    ),
    "class12_themes_in_indian_history_3": (
        "zip",
        "https://ncert.nic.in/textbook/pdf/lehs3dd.zip",
        "12",
    ),
    # Old NCERT (public-domain-style, denser prose -- good "hard mode" corpus)
    "old_ncert_ancient_india_rs_sharma": (
        "pdf",
        "https://vajiramias.sgp1.cdn.digitaloceanspaces.com/strapi/media/2023/8/18/17/3/43/01_Old_NCERT_Ancient_India.pdf",
        "old-ancient",
    ),
    "old_ncert_medieval_india_satish_chandra": (
        "pdf",
        "https://vajiramias.sgp1.cdn.digitaloceanspaces.com/strapi/media/2023/8/18/17/2/56/01_Old_NCERT_Medieval_India.pdf",
        "old-medieval",
    ),
    "old_ncert_modern_india_bipan_chandra": (
        "pdf",
        "https://vajiramias.sgp1.cdn.digitaloceanspaces.com/strapi/media/2023/8/18/17/1/51/old-ncert-books-for-upsc-Modern-India-Bipan-Chandra.pdf",
        "old-modern",
    ),
}

OUTPUT_DIR = "ncert_history_corpus"

# A generic bot-like User-Agent gets silently blocked/throttled by some
# government sites (this was the actual cause of the lehs1dd.zip failure).
# Using a real browser UA + a Referer fixes it in almost all cases.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://ncert.nic.in/textbook.php",
    "Accept": "*/*",
}
TIMEOUT = 90
RETRIES = 5


def download_file(url: str, dest_path: str) -> bool:
    session = requests.Session()
    session.headers.update(HEADERS)

    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            content_type = resp.headers.get("Content-Type", "unknown")

            if resp.status_code == 200 and len(resp.content) > 1000:
                # Sanity check: a blocked/soft-404 request often comes
                # back as a small HTML page with a 200 status instead
                # of a real file -- catch that case explicitly.
                if b"<html" in resp.content[:200].lower():
                    print(f"    [attempt {attempt}] got an HTML page instead "
                          f"of a file (likely blocked) for {url}")
                else:
                    with open(dest_path, "wb") as f:
                        f.write(resp.content)
                    return True
            else:
                print(f"    [attempt {attempt}] bad response "
                      f"(status={resp.status_code}, len={len(resp.content)}, "
                      f"content-type={content_type}) for {url}")
        except requests.RequestException as e:
            print(f"    [attempt {attempt}] error: {e}")

        if attempt < RETRIES:
            wait = 3 * attempt  # gentle backoff: 3s, 6s, 9s, 12s...
            print(f"    retrying in {wait}s...")
            time.sleep(wait)

    return False


def unzip_book(zip_path: str, out_folder: str) -> list:
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.lower().endswith(".pdf"):
                z.extract(name, out_folder)
                extracted.append(name)
    return extracted


def file_hash(path: str) -> str:
    """MD5 of file content, in chunks so large PDFs don't blow up memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_existing_hashes(root: str) -> dict:
    """
    Walk the whole corpus folder and hash every PDF already there,
    regardless of which subfolder it's in. This is what catches a
    case like a manually-extracted 'lehs1dd' folder duplicating what
    the script would otherwise download into 'class12_..._1'.
    Returns {hash: path}.
    """
    hashes = {}
    if not os.path.isdir(root):
        return hashes
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".pdf"):
                full = os.path.join(dirpath, fname)
                try:
                    hashes[file_hash(full)] = full
                except OSError:
                    pass
    return hashes


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = []

    print("Scanning existing corpus for duplicates (this may take a moment)...")
    known_hashes = scan_existing_hashes(OUTPUT_DIR)
    print(f"Found {len(known_hashes)} existing PDF(s) already in the corpus.\n")

    for label, (kind, url, class_level) in BOOKS.items():
        print(f"=== {label} (class {class_level}) ===")
        book_folder = os.path.join(OUTPUT_DIR, label)
        os.makedirs(book_folder, exist_ok=True)

        if kind == "pdf":
            dest = os.path.join(book_folder, f"{label}.pdf")

            if os.path.exists(dest):
                print(f"  Already downloaded -> {dest} (skipping)")
                manifest.append({"book": label, "class": class_level,
                                  "file": os.path.basename(dest), "source_url": url})
                continue

            print(f"  Downloading {url} ...")
            if not download_file(url, dest):
                print(f"  FAILED to download {label}.")
                continue

            h = file_hash(dest)
            if h in known_hashes:
                print(f"  Duplicate of {known_hashes[h]} -- removing this copy.")
                os.remove(dest)
                continue

            known_hashes[h] = dest
            print(f"  Saved -> {dest}")
            manifest.append({"book": label, "class": class_level,
                              "file": os.path.basename(dest), "source_url": url})

        elif kind == "zip":
            # If this book's folder already has chapter PDFs, assume it's
            # already been downloaded+extracted (covers a normal re-run).
            existing_pdfs = [f for f in os.listdir(book_folder)
                              if f.lower().endswith(".pdf")]
            if existing_pdfs:
                print(f"  Already extracted ({len(existing_pdfs)} chapter PDFs "
                      f"in {book_folder}) -- skipping download.")
                for ch in sorted(existing_pdfs):
                    manifest.append({"book": label, "class": class_level,
                                      "file": ch, "source_url": url})
                continue

            zip_path = os.path.join(book_folder, f"{label}.zip")
            print(f"  Downloading {url} ...")
            if not download_file(url, zip_path):
                print(f"  FAILED to download {label}.")
                continue
            try:
                chapters = unzip_book(zip_path, book_folder)
            except zipfile.BadZipFile:
                print(f"  FAILED: downloaded file for {label} is not a valid ZIP "
                      f"(the NCERT link may have changed).")
                os.remove(zip_path)
                continue
            os.remove(zip_path)
            if not chapters:
                print(f"  Downloaded but found no PDFs inside the ZIP for {label}.")
                continue

            kept = []
            for ch in sorted(chapters):
                ch_path = os.path.join(book_folder, ch)
                h = file_hash(ch_path)
                if h in known_hashes:
                    print(f"  Skipping {ch}: duplicate of {known_hashes[h]}")
                    os.remove(ch_path)
                    continue
                known_hashes[h] = ch_path
                kept.append(ch)
                manifest.append({"book": label, "class": class_level,
                                  "file": ch, "source_url": url})

            print(f"  Extracted {len(kept)} new chapter PDF(s) "
                  f"({len(chapters) - len(kept)} duplicate(s) skipped) -> {book_folder}/")

        print()

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("book,class,file,source_url\n")
        for row in manifest:
            f.write(f"{row['book']},{row['class']},{row['file']},{row['source_url']}\n")

    print(f"Done. {len(manifest)} PDF files total in manifest.")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()