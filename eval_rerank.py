#!/usr/bin/env python3
"""
Cross-encoder reranking over the retrieval candidates.

WHY A RERANKER, AND ITS CEILING
-------------------------------
Measured headroom on this corpus (recall@50 minus recall@5):

    dense   0.63 -> 0.80   headroom 0.16
    bm25    0.73 -> 0.84   headroom 0.10

A reranker only reorders what retrieval returned, so those numbers are the
most a PERFECT reranker could add. Worth building, but it also means ~16% of
questions have no gold chunk anywhere in the top 50 and are untouchable by
reranking -- those are coverage problems (wrong gold, bad chunk boundary, or
vocabulary the query and corpus do not share).

WHY FUSION STILL APPEARS HERE AFTER THE SWEEP SAID NO
-----------------------------------------------------
The alpha/k_rrf sweep found no fusion setting beating BM25 alone at k=5, so
fusion does not help RANKING. Coverage is a separate question: if dense and
BM25 miss different questions, the union of their top-50 lists is a better
candidate pool than either, even though fusing their rankings was useless.
This script therefore takes the UNION as candidates and lets the cross-encoder
do the ordering -- which is the job fusion was failing at.

THE MODEL
---------
cross-encoder/ms-marco-MiniLM-L-6-v2: scores (query, passage) pairs jointly
rather than embedding them separately, so it can use term overlap and context
a bi-encoder averages away. Slower -- it runs the model once per candidate --
which is why it reranks 50 candidates rather than 2,859 chunks.

USAGE
-----
    pip install sentence-transformers          # cross-encoders need this
    python eval_rerank.py
    python eval_rerank.py --candidates 30 --failures
"""

import os
import re
import json
import argparse
from collections import defaultdict

DEFAULT_DB = "chroma_db"
DEFAULT_COLLECTION = "chunks__bge_small_en_v1_5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TIER_ORDER = ["easy", "medium", "hard", "trick"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def get_query_encoder(model_name, prefix, backend="auto"):
    def wrap(fn):
        return lambda texts: fn([prefix + t for t in texts])
    if backend in ("auto", "sentence-transformers"):
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(model_name)
            return wrap(lambda ts: m.encode(ts, normalize_embeddings=True,
                                            convert_to_numpy=True,
                                            show_progress_bar=False).tolist())
        except ImportError:
            if backend == "sentence-transformers":
                raise
            print("sentence-transformers unavailable, using fastembed")
    from fastembed import TextEmbedding
    m = TextEmbedding(model_name=model_name)
    return wrap(lambda ts: [v.tolist() for v in m.embed(ts)])


def score_one(ranked, gold, mode, k):
    top = ranked[:k]
    hits = [i for i, c in enumerate(top) if c in gold]
    first = hits[0] + 1 if hits else None
    full = (1.0 if gold.issubset(set(top)) else 0.0) if mode == "all" \
        else (1.0 if hits else 0.0)
    return {"recall_at_k": 1.0 if hits else 0.0, "full_at_k": full,
            "mrr": 1.0 / first if first else 0.0, "first_rank": first,
            "retrieved": top}


def table(all_results, k):
    methods = list(all_results.keys())
    print(f"\n{'='*78}")
    print(f"RERANKING @ k={k}")
    print("=" * 78)
    print(f"{'tier':8s} {'n':>3s} " + " ".join(f"| {m:^17s}" for m in methods))
    print(f"{'':8s} {'':>3s} " + " ".join(f"| {'rec  full  mrr':^17s}"
                                          for _ in methods))
    print("-" * 78)
    for tier in TIER_ORDER + ["ALL"]:
        rows = {m: [r for r in all_results[m]
                    if tier == "ALL" or r["tier"] == tier] for m in methods}
        n = len(rows[methods[0]])
        if not n:
            continue
        line = f"{tier:8s} {n:3d} "
        for m in methods:
            rs = rows[m]
            line += (f"| {sum(x['recall_at_k'] for x in rs)/n:4.2f} "
                     f"{sum(x['full_at_k'] for x in rs)/n:5.2f} "
                     f"{sum(x['mrr'] for x in rs)/n:5.2f} ")
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval_set.json")
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=50,
                    help="how many candidates from EACH retriever to rerank")
    ap.add_argument("--model", default=RERANK_MODEL)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--out", default="eval_rerank_results.json")
    args = ap.parse_args()

    for f in (args.eval_set, args.chunks):
        if not os.path.exists(f):
            print(f"ERROR: {f} not found.")
            return

    raw = json.load(open(args.eval_set, encoding="utf-8"))
    eval_set = raw.get("questions", raw) if isinstance(raw, dict) else raw
    eval_source = raw.get("source_file", "") if isinstance(raw, dict) else ""

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8")
              if l.strip()]
    text_of = {c["chunk_id"]: c["text"] for c in chunks}

    if eval_source and os.path.basename(args.chunks) != eval_source:
        print(f"ERROR: eval set built from '{eval_source}', --chunks is "
              f"'{os.path.basename(args.chunks)}'. Not comparable.")
        return

    import chromadb
    client = chromadb.PersistentClient(path=args.db)
    try:
        coll = client.get_collection(args.collection)
    except Exception:
        print(f"ERROR: collection not found. Available: "
              f"{[c.name for c in client.list_collections()]}")
        return

    meta = coll.metadata or {}
    if eval_source and meta.get("source_file") and \
            os.path.basename(meta["source_file"]) != eval_source:
        print(f"ERROR: eval set from '{eval_source}', collection from "
              f"'{meta['source_file']}'. Not comparable.")
        return

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("ERROR: pip install rank-bm25")
        return
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("ERROR: cross-encoders need sentence-transformers.")
        print("  pip install sentence-transformers")
        print("If it fails to import on this machine (blocked sklearn DLL), "
              "reranking is not available via fastembed and this step must "
              "run elsewhere.")
        return

    print(f"Collection: {args.collection} ({coll.count()} chunks)")
    print(f"Building BM25 index over {len(chunks)} chunks...")
    ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    questions = [e for e in eval_set if e.get("gold_chunk_ids")]
    print(f"Loading reranker {args.model}...")
    ce = CrossEncoder(args.model)

    encode = get_query_encoder(meta.get("model", "BAAI/bge-small-en-v1.5"),
                               meta.get("query_prefix", ""), args.backend)
    qvecs = encode([e["question"] for e in questions])

    results = {"bm25": [], "dense": [], "union+rerank": []}
    print(f"Reranking {len(questions)} questions "
          f"({args.candidates} candidates each)...")

    for n_done, (e, v) in enumerate(zip(questions, qvecs), 1):
        gold = set(e["gold_chunk_ids"])
        mode = e.get("gold_mode", "all" if e.get("multi_chunk") else "any")

        dense_ranked = coll.query(query_embeddings=[v],
                                  n_results=args.candidates,
                                  include=[])["ids"][0]
        sc = bm25.get_scores(tokenize(e["question"]))
        bm25_ranked = [ids[i] for i in
                       sorted(range(len(ids)), key=lambda i: -sc[i])
                       [:args.candidates]]

        # Union as the candidate pool: the sweep showed fusing RANKINGS does
        # not help, but the union is about COVERAGE, and the cross-encoder
        # supplies the ordering.
        pool = list(dict.fromkeys(dense_ranked + bm25_ranked))
        pairs = [(e["question"], text_of.get(cid, "")) for cid in pool]
        scores = ce.predict(pairs, show_progress_bar=False)
        reranked = [cid for cid, _ in
                    sorted(zip(pool, scores), key=lambda x: -x[1])]

        base = {"id": e["id"], "tier": e["tier"], "question": e["question"],
                "verified": e.get("verified", False), "gold_mode": mode,
                "gold_chunk_ids": sorted(gold), "note": e.get("note", ""),
                "pool_size": len(pool)}
        for name, ranked in (("bm25", bm25_ranked), ("dense", dense_ranked),
                             ("union+rerank", reranked)):
            row = dict(base)
            row.update(score_one(ranked, gold, mode, args.k))
            results[name].append(row)

        if n_done % 10 == 0:
            print(f"  {n_done}/{len(questions)}")

    table(results, args.k)

    # what reranking changed, per question
    b = {r["id"]: r for r in results["bm25"]}
    rr = {r["id"]: r for r in results["union+rerank"]}
    gained = [q for q in b if b[q]["recall_at_k"] == 0
              and rr[q]["recall_at_k"] == 1]
    lost = [q for q in b if b[q]["recall_at_k"] == 1
            and rr[q]["recall_at_k"] == 0]
    print(f"\nvs BM25 alone:  gained {len(gained)} {gained}")
    print(f"                lost   {len(lost)} {lost}")
    if not gained and lost:
        print("  Reranking is a net loss here. Report that rather than tuning "
              "until it looks good.")

    if args.failures:
        print(f"\n{'='*78}\nSTILL FAILING AFTER RERANK\n{'='*78}")
        for r in results["union+rerank"]:
            if r["recall_at_k"] == 0.0:
                in_pool = set(r["gold_chunk_ids"]) & set(r["retrieved"])
                print(f"\n[{r['id']}] {r['tier']}  pool={r['pool_size']}")
                print(f"  Q: {r['question'][:84]}")
                print(f"  gold: {', '.join(r['gold_chunk_ids'][:3])}")
                print(f"  gold was in candidate pool: "
                      f"{'yes -- ranking problem' if in_pool else 'no -- coverage problem'}")
                if r["note"]:
                    print(f"  note: {r['note'][:100]}")

    json.dump({"collection": args.collection, "k": args.k,
               "candidates": args.candidates, "rerank_model": args.model,
               "results": results},
              open(args.out, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"\nDetail -> {args.out}")


if __name__ == "__main__":
    main()