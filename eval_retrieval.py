#!/usr/bin/env python3
"""
Score retrieval quality against the evaluation set.

WHAT IT MEASURES
----------------
Retrieval only. Generation is scored separately, later, and deliberately so:
a wrong final answer can mean the right chunk was never retrieved, or that it
was retrieved and the model ignored it. Those need different fixes, and a
single end-to-end accuracy number cannot tell them apart.

Metrics, reported per tier rather than as one aggregate:

  recall@k   did ANY gold chunk appear in the top k?
  full@k     did ALL gold chunks appear in the top k?  (the honest metric for
             multi-hop questions -- recall@k rewards finding one of two
             required chunks, which is not success)
  MRR        1 / rank of the first gold chunk, 0 if absent

A flat aggregate hides exactly what matters here. Easy questions should be
near-perfect; if they are not, something is broken and no reranking will save
it. The interesting signal is the gap between tiers.

HONESTY ABOUT GOLDS
-------------------
Questions whose `verified` flag is false are scored, but counted and reported
separately. An unverified gold is a search hit somebody has not confirmed, and
numbers computed from it are provisional.

USAGE
-----
  python eval_retrieval.py
  python eval_retrieval.py --collection chunks_nosections__bge_small_en_v1_5
  python eval_retrieval.py --k 10 --failures    # print failure detail
"""

import os
import json
import argparse
from collections import defaultdict

DEFAULT_DB = "chroma_db"
DEFAULT_COLLECTION = "chunks__bge_small_en_v1_5"
TIER_ORDER = ["easy", "medium", "hard", "trick"]


def get_query_encoder(model_name, prefix, backend="auto"):
    """
    Returns a function that embeds QUERY text.

    BGE is asymmetric: queries need an instruction prefix, passages do not.
    The prefix comes from the collection metadata rather than being hard-coded,
    so ingestion and retrieval cannot drift apart.
    """
    def wrap(encode_fn):
        def encode_queries(texts):
            return encode_fn([prefix + t for t in texts])
        return encode_queries

    if backend in ("auto", "sentence-transformers"):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            return wrap(lambda ts: model.encode(
                ts, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False).tolist())
        except ImportError as e:
            if backend == "sentence-transformers":
                raise
            print(f"sentence-transformers unavailable ({e.__class__.__name__}), "
                  f"falling back to fastembed")

    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=model_name)
    return wrap(lambda ts: [v.tolist() for v in model.embed(ts)])


def evaluate(eval_set, coll, encode_queries, k):
    results = []
    questions = [e for e in eval_set if e.get("gold_chunk_ids")]
    skipped = [e["id"] for e in eval_set if not e.get("gold_chunk_ids")]

    if not questions:
        print("No questions have gold_chunk_ids set. Fill them in first.")
        return [], skipped

    vecs = encode_queries([e["question"] for e in questions])

    for e, v in zip(questions, vecs):
        res = coll.query(query_embeddings=[v], n_results=k,
                         include=["metadatas", "distances"])
        retrieved = res["ids"][0]
        gold = set(e["gold_chunk_ids"])

        # "any" -> any gold chunk suffices (several chunks may state the same
        # fact). "all" -> a multi-hop question needing every gold chunk.
        mode = e.get("gold_mode", "all" if e.get("multi_chunk") else "any")
        hits = [i for i, cid in enumerate(retrieved) if cid in gold]
        first_rank = hits[0] + 1 if hits else None
        found = {retrieved[i] for i in hits}

        results.append({
            "id": e["id"],
            "tier": e["tier"],
            "question": e["question"],
            "expected_behaviour": e.get("expected_behaviour", "answer"),
            "multi_chunk": e.get("multi_chunk", False),
            "verified": e.get("verified", False),
            "n_gold": len(gold),
            "recall_at_k": 1.0 if hits else 0.0,
            "full_at_k": (1.0 if gold.issubset(set(retrieved)) else 0.0)
                          if mode == "all"
                          else (1.0 if hits else 0.0),
            "gold_mode": mode,
            "mrr": 1.0 / first_rank if first_rank else 0.0,
            "first_rank": first_rank,
            "gold_found": sorted(found),
            "gold_missed": sorted(gold - found),
            "retrieved": retrieved,
            "note": e.get("note", ""),
        })
    return results, skipped


def summarise(results, k, label):
    print(f"\n{'='*66}")
    print(f"RETRIEVAL @ k={k}   [{label}]")
    print("=" * 66)

    by_tier = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)

    print(f"{'tier':9s} {'n':>3s} {'recall@k':>9s} {'full@k':>8s} "
          f"{'MRR':>6s} {'unverified':>11s}")
    print("-" * 66)
    for tier in TIER_ORDER:
        rs = by_tier.get(tier, [])
        if not rs:
            continue
        n = len(rs)
        unv = sum(1 for r in rs if not r["verified"])
        print(f"{tier:9s} {n:3d} "
              f"{sum(r['recall_at_k'] for r in rs)/n:9.2f} "
              f"{sum(r['full_at_k'] for r in rs)/n:8.2f} "
              f"{sum(r['mrr'] for r in rs)/n:6.2f} "
              f"{unv:11d}")

    n = len(results)
    unv = sum(1 for r in results if not r["verified"])
    print("-" * 66)
    print(f"{'ALL':9s} {n:3d} "
          f"{sum(r['recall_at_k'] for r in results)/n:9.2f} "
          f"{sum(r['full_at_k'] for r in results)/n:8.2f} "
          f"{sum(r['mrr'] for r in results)/n:6.2f} "
          f"{unv:11d}")

    if unv:
        print(f"\n{unv} of {n} questions have UNVERIFIED gold chunks. "
              f"Those rows are provisional.")

    # multi-chunk questions: recall@k flatters them, full@k does not
    multi = [r for r in results if r["multi_chunk"]]
    if multi:
        mr = sum(r["recall_at_k"] for r in multi) / len(multi)
        mf = sum(r["full_at_k"] for r in multi) / len(multi)
        print(f"\nMulti-chunk questions ({len(multi)}): "
              f"recall@k {mr:.2f} vs full@k {mf:.2f}")
        if mr - mf > 0.15:
            print("  The gap is the point: partial retrieval looks like success "
                  "under recall@k but cannot answer the question.")

    # the corpus-defect questions
    defects = [r for r in results if r["id"] in ("X1", "X2")]
    if defects:
        print("\nOCR-defect questions:")
        for r in defects:
            status = f"rank {r['first_rank']}" if r["first_rank"] else "NOT FOUND"
            print(f"  {r['id']}: {status}")


def print_failures(results, k):
    fails = [r for r in results if r["recall_at_k"] == 0.0]
    partial = [r for r in results if r["recall_at_k"] == 1.0
               and r["full_at_k"] == 0.0]

    print(f"\n{'='*66}")
    print(f"FAILURES ({len(fails)} total miss, {len(partial)} partial)")
    print("=" * 66)

    for r in fails:
        print(f"\n[{r['id']}] {r['tier']} -- no gold chunk in top {k}")
        print(f"  Q: {r['question'][:88]}")
        print(f"  gold:      {', '.join(r['gold_missed'])}")
        print(f"  retrieved: {', '.join(r['retrieved'][:3])}")
        if r["note"]:
            print(f"  note: {r['note'][:110]}")

    for r in partial:
        print(f"\n[{r['id']}] {r['tier']} -- partial "
              f"({len(r['gold_found'])}/{r['n_gold']} gold chunks)")
        print(f"  Q: {r['question'][:88]}")
        print(f"  missed: {', '.join(r['gold_missed'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval_set.json")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "fastembed", "sentence-transformers"])
    ap.add_argument("--force", action="store_true",
                    help="score even if the eval set and collection came from "
                         "different chunk files (results will be meaningless)")
    ap.add_argument("--failures", action="store_true",
                    help="print per-question failure detail")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    if not os.path.exists(args.eval_set):
        print(f"ERROR: {args.eval_set} not found. Run backfill_gold_chunks.py "
              f"and fill in gold_chunk_ids first.")
        return

    raw = json.load(open(args.eval_set, encoding="utf-8"))
    # Support both the wrapped format and a bare list (older files).
    if isinstance(raw, dict):
        eval_set = raw.get("questions", [])
        eval_source = raw.get("source_file", "")
    else:
        eval_set, eval_source = raw, ""

    import chromadb
    client = chromadb.PersistentClient(path=args.db)
    try:
        coll = client.get_collection(args.collection)
    except Exception:
        names = [c.name for c in client.list_collections()]
        print(f"ERROR: collection '{args.collection}' not found.")
        print(f"Available: {names}")
        return

    meta = coll.metadata or {}
    model = meta.get("model", "BAAI/bge-small-en-v1.5")
    prefix = meta.get("query_prefix", "")
    print(f"Collection: {args.collection} ({coll.count()} chunks)")
    print(f"Model: {model}")
    print(f"Query prefix: {prefix!r}"
          + ("  <-- empty; BGE needs one, check ingestion"
             if not prefix and "bge" in model.lower() else ""))

    # Guard against scoring golds against the wrong collection.
    #
    # Chunk IDs are sequential per book, so the sections and no-sections runs
    # reuse the same ID strings for different text. Scoring one against the
    # other produces plausible-looking numbers that measure nothing, and
    # nothing errors. This check exists because that actually happened.
    coll_source = meta.get("source_file", "")
    if eval_source and coll_source and \
            os.path.basename(coll_source) != eval_source:
        print(f"\nERROR: gold chunk IDs were generated from '{eval_source}' "
              f"but this collection was built from '{coll_source}'.")
        print("Chunk IDs are not comparable across chunking runs -- the same "
              "ID refers to different text.")
        print(f"Either score against the collection built from "
              f"'{eval_source}', or re-run backfill_gold_chunks.py "
              f"--input {coll_source} to generate golds for this one.")
        if not args.force:
            return
        print("--force given; continuing anyway. Results are not meaningful.")

    encode_queries = get_query_encoder(model, prefix, args.backend)
    results, skipped = evaluate(eval_set, coll, encode_queries, args.k)
    if not results:
        return

    if skipped:
        print(f"\nSkipped {len(skipped)} question(s) with no gold chunks set: "
              f"{', '.join(skipped)}")

    summarise(results, args.k, args.collection)
    if args.failures:
        print_failures(results, args.k)

    json.dump({"collection": args.collection, "model": model, "k": args.k,
               "results": results},
              open(args.out, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"\nPer-question detail -> {args.out}")
    print("\nRun again with --collection chunks_nosections__bge_small_en_v1_5 "
          "to compare against the no-sections chunking.")


if __name__ == "__main__":
    main()