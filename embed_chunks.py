#!/usr/bin/env python3
"""
Embed chunks.jsonl and load them into a local Chroma collection.

MODEL CHOICE
------------
Default is BAAI/bge-small-en-v1.5 rather than the more commonly used
all-MiniLM-L6-v2.

Reason: MiniLM truncates at 256 word-pieces. Chunks here are packed to ~400
estimated tokens with a 90th percentile near 399, so MiniLM would silently
discard the tail of a large fraction of chunks -- silently being the problem,
since nothing errors and retrieval just quietly gets worse. bge-small handles
512 tokens, is a similar size (~33M params), and stays CPU-viable.

BGE models are asymmetric: passages are embedded as-is, but QUERIES need an
instruction prefix. That prefix is written into the collection metadata by this
script so the retrieval code can read it back rather than hard-coding it in two
places and drifting.

IDEMPOTENCE
-----------
Re-running skips chunks already present in the collection, so an interrupted
run resumes instead of starting over or double-inserting.

A/B COMPARISON
--------------
Collection names encode the config (model + input file), so the section-aware
and no-sections chunk sets land in separate collections and can be evaluated
against each other:

    python embed_chunks.py --input chunks.jsonl
    python embed_chunks.py --input chunks_nosections.jsonl

USAGE
-----
    pip install sentence-transformers chromadb
    python embed_chunks.py
    python embed_chunks.py --model sentence-transformers/all-MiniLM-L6-v2
    python embed_chunks.py --limit 50        # smoke test before the full run
"""

import os
import re
import json
import time
import argparse

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_INPUT = "chunks.jsonl"
DEFAULT_DB = "chroma_db"

# BGE-family models need this on the QUERY side only, not on passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chroma rejects None in metadata, so missing values get a sentinel instead.
MISSING_INT = -1
MISSING_STR = ""


def collection_name(model: str, input_path: str) -> str:
    """Encode the config in the name so A/B runs don't overwrite each other."""
    model_tag = re.sub(r"[^a-z0-9]+", "_", model.lower().split("/")[-1]).strip("_")
    input_tag = re.sub(r"[^a-z0-9]+", "_",
                       os.path.splitext(os.path.basename(input_path))[0].lower())
    name = f"{input_tag}__{model_tag}"
    # Chroma requires 3-512 chars, start/end alphanumeric
    return name[:500]


def sanitise_metadata(chunk: dict) -> dict:
    """
    Chroma metadata values must be str / int / float / bool -- None is
    rejected. 148 chunks legitimately have no page number (their chapter slice
    begins before the first page marker), so those become MISSING_INT rather
    than being dropped or crashing the load.
    """
    def s(v):
        return MISSING_STR if v is None else str(v)

    def i(v):
        if v is None or v == "":
            return MISSING_INT
        try:
            return int(v)
        except (TypeError, ValueError):
            return MISSING_INT

    return {
        "book": s(chunk.get("book")),
        "chapter_num": i(chunk.get("chapter_num")),
        "chapter_title": s(chunk.get("chapter_title")),
        "chapter_confidence": s(chunk.get("chapter_confidence")),
        "section": s(chunk.get("section")),
        "page_start": i(chunk.get("page_start")),
        "page_end": i(chunk.get("page_end")),
        "content_type": s(chunk.get("content_type")),
        "source_type": s(chunk.get("source_type")),
        "n_tokens_est": i(chunk.get("n_tokens_est")),
    }


def load_chunks(path, limit=None):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
            if limit and len(chunks) >= limit:
                break
    return chunks


def get_embedder_fastembed(model_name):
    """
    ONNX-runtime backend. Same model weights, but avoids the
    sentence-transformers -> sklearn -> torch dependency chain.

    Needed because sentence-transformers imports sklearn at module load (for a
    cosine-similarity helper it does not even need here), and on some Windows
    machines the OS blocks sklearn's compiled BLAS DLL:

        ImportError: DLL load failed while importing _cython_blas:
        An Application Control policy has blocked this file.

    That is a machine policy, not a code fault. fastembed sidesteps it.
    """
    from fastembed import TextEmbedding
    print(f"Loading {model_name} via fastembed/ONNX "
          f"(first run downloads weights)...")
    model = TextEmbedding(model_name=model_name)

    def encode(texts, batch_size=32):
        # fastembed returns a generator of numpy arrays, already normalised.
        return [v.tolist() for v in model.embed(texts, batch_size=batch_size)]

    dim = len(encode(["dimension probe"])[0])
    print(f"  dim={dim}")
    return encode, dim, None


def get_embedder(model_name):
    """
    Returns (encode_fn, dim, max_seq). Isolated so the rest of the pipeline can
    be tested without downloading model weights.
    """
    from sentence_transformers import SentenceTransformer
    print(f"Loading model {model_name} (first run downloads weights)...")
    model = SentenceTransformer(model_name)
    # Renamed in newer sentence-transformers; support both without warning.
    dim = (model.get_embedding_dimension()
           if hasattr(model, "get_embedding_dimension")
           else model.get_sentence_embedding_dimension())
    max_seq = getattr(model, "max_seq_length", None)
    print(f"  dim={dim}, max_seq_length={max_seq}")

    def encode(texts, batch_size=32):
        # normalize_embeddings=True so cosine similarity == dot product,
        # which is what the collection is configured for.
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    return encode, dim, max_seq


def main(encode_override=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="only embed the first N chunks (smoke test)")
    ap.add_argument("--reset", action="store_true",
                    help="delete the collection first")
    ap.add_argument("--backend", choices=["auto", "fastembed",
                                          "sentence-transformers"],
                    default="auto",
                    help="embedding backend; 'auto' tries sentence-transformers "
                         "then falls back to fastembed if its import fails")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run chunk_corpus.py first.")
        return

    chunks = load_chunks(args.input, args.limit)
    print(f"Loaded {len(chunks)} chunks from {args.input}")

    import chromadb
    client = chromadb.PersistentClient(path=args.db)
    cname = collection_name(args.model, args.input)

    if args.reset:
        try:
            client.delete_collection(cname)
            print(f"Deleted existing collection {cname}")
        except Exception:
            pass

    # Embeddings are normalised, so cosine is the right space.
    coll = client.get_or_create_collection(
        name=cname,
        metadata={
            "hnsw:space": "cosine",
            "model": args.model,
            "source_file": args.input,
            # Retrieval code reads this instead of hard-coding the prefix.
            "query_prefix": BGE_QUERY_PREFIX if "bge" in args.model.lower() else "",
        },
    )
    print(f"Collection: {cname} (existing count: {coll.count()})")

    # --- idempotence: skip what's already in there ---
    existing = set()
    if coll.count() > 0:
        got = coll.get(include=[], limit=coll.count())
        existing = set(got["ids"])
        print(f"  {len(existing)} chunks already embedded, will skip those")

    todo = [c for c in chunks if c["chunk_id"] not in existing]
    if not todo:
        print("Nothing to do -- all chunks already embedded.")
        report(coll)
        return
    print(f"  {len(todo)} chunks to embed")

    if encode_override is not None:
        encode, dim, max_seq = encode_override, None, None
    elif args.backend == "fastembed":
        encode, dim, max_seq = get_embedder_fastembed(args.model)
    elif args.backend == "sentence-transformers":
        encode, dim, max_seq = get_embedder(args.model)
    else:
        try:
            encode, dim, max_seq = get_embedder(args.model)
        except ImportError as e:
            print(f"sentence-transformers unavailable ({e.__class__.__name__}: "
                  f"{str(e)[:90]})")
            print("Falling back to fastembed/ONNX backend...")
            encode, dim, max_seq = get_embedder_fastembed(args.model)

    # --- truncation check ---------------------------------------------
    # n_tokens_est comes from a words*1.3 heuristic. Before embedding 2,859
    # chunks, verify against the model's REAL tokenizer that nothing is being
    # silently cut. Truncation does not raise -- it just quietly discards the
    # tail of a chunk, so retrieval degrades with no visible symptom.
    if max_seq is not None and encode_override is None:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
            sample = todo[:400]
            lens = [len(tok.encode(c["text"], add_special_tokens=True))
                    for c in sample]
            over = [l for l in lens if l > max_seq]
            print(f"  tokenizer check on {len(sample)} chunks: "
                  f"max {max(lens)} tokens, limit {max_seq}, "
                  f"{len(over)} would be truncated")
            if over:
                print(f"  WARNING: {len(over)} chunk(s) exceed the model limit "
                      f"and will lose their tail. Consider re-chunking with a "
                      f"smaller --target-tokens.")
        except Exception as e:
            print(f"  (tokenizer check skipped: {e.__class__.__name__})")

    t0 = time.time()
    done = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        texts = [c["text"] for c in batch]          # passages: no prefix
        vecs = encode(texts, batch_size=args.batch_size) \
            if encode_override is None else encode(texts)

        coll.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vecs,
            documents=texts,
            metadatas=[sanitise_metadata(c) for c in batch],
        )
        done += len(batch)
        if done % (args.batch_size * 10) == 0 or done == len(todo):
            rate = done / max(1e-9, time.time() - t0)
            eta = (len(todo) - done) / max(1e-9, rate)
            print(f"    {done}/{len(todo)} chunks "
                  f"({rate:.1f}/s, eta {eta/60:.1f} min)")

    print(f"\nEmbedded {done} chunks in {(time.time()-t0)/60:.1f} min")
    report(coll)


def report(coll):
    n = coll.count()
    print(f"\nCollection now holds {n} chunks.")
    if n == 0:
        return
    got = coll.get(include=["metadatas"], limit=n)
    from collections import Counter
    for field in ("source_type", "content_type"):
        c = Counter(m.get(field) for m in got["metadatas"])
        print(f"  {field}: {dict(c.most_common(5))}")
    books = Counter(m.get("book") for m in got["metadatas"])
    print(f"  books: {len(books)}")
    missing_pages = sum(1 for m in got["metadatas"]
                        if m.get("page_start") == MISSING_INT)
    print(f"  chunks without a page number: {missing_pages} "
          f"(stored as {MISSING_INT}, not dropped)")
    print("\nNext: build the retrieval + eval harness. Use the collection's "
          "`query_prefix` metadata when embedding queries -- BGE is asymmetric "
          "and omitting it measurably degrades retrieval.")


if __name__ == "__main__":
    main()