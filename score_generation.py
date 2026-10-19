#!/usr/bin/env python3
"""
Score generation_results.json (output of generate_answers.py).

Retrieval and generation are scored separately. Three layers, cheapest first:

1. DETERMINISTIC checks (free, no API):
   behaviour   status == expected_behaviour (answer / abstain / clarify)
   citations   every cited ID (field and inline [id]) was actually retrieved;
               a cited ID that was never shown to the model is fabricated
   gold_in_ctx did retrieval put the gold chunk(s) in front of the model?
               (separates "model failed" from "retrieval failed")
   unsupported numbers and capitalised words in the answer that appear in NONE
               of the retrieved passages: a cheap hallucination tripwire
               (noisy: paraphrase and OCR variants cause false alarms)

2. LLM JUDGES (one API call per answered question per judge): split the
   answer into atomic claims and label each supported / unsupported, quoting
   the passage sentence that supports it. groundedness = supported / total.

   Bias controls:
   - Independent primary judge. Default judges are claude-opus-5-5 (primary,
     a different model from the generator) and claude-sonnet-5-5 (the
     generator judging itself). The headline number comes from the primary.
   - Self-preference is MEASURED, not assumed: the gap between the self-judge
     and the independent judge on the same answers, plus their agreement.
   - Evidence must be checkable: a "supported" verdict whose quote cannot be
     found in the passages is downgraded to "unverified" (counted as not
     supported). A judge cannot approve a claim on a feeling.
   Remaining limit: both judges are Claude models, so blind spots shared by
   the model family are reduced, not removed.

3. MANUAL SPOT-CHECK: writes spot_check.md with a seeded sample of answers
   next to their passages, for a human to verify. Report how many you checked
   and how often you agreed with the judge. That agreement rate is what makes
   the judge number credible.

NOT scored: answer correctness. eval_set.json stores gold chunks, not gold
answer text, so "is the answer right" is only approximated by gold_in_ctx +
groundedness.

USAGE
    python score_generation.py                  # all layers
    python score_generation.py --no-judge       # deterministic only, free
    python score_generation.py --judges claude-opus-5-5     # one judge only
    python score_generation.py --spot 12 --seed 1
"""

import re
import sys
import json
import random
import difflib
import argparse
from collections import defaultdict

from generate_answers import (DEFAULT_MODEL, call_claude, make_client, parse_json)

TIER_ORDER = ["easy", "medium", "hard", "trick"]

JUDGE_SYSTEM = """You are a strict fact-checking judge. You are given PASSAGES and an ANSWER. You do not know who or what wrote the answer; judge it only against the passages.

1. Split the ANSWER into atomic factual claims (ignore citations, hedges and meta-statements like "the passages do not say").
2. For each claim decide whether the PASSAGES, read allowing for OCR spelling errors, directly support it.
   - "supported": stated in or clearly entailed by the passages
   - "unsupported": not stated in the passages (even if true in the real world), or contradicted by them, or an interpretation the passages do not make
3. For every "supported" claim, copy into "evidence" the exact sentence or phrase from the PASSAGES that supports it, character for character (OCR errors included). For "unsupported" claims, leave "evidence" empty.

Respond with a single JSON object and nothing else:
{"claims": [{"claim": "<text>", "verdict": "supported" | "unsupported", "evidence": "<exact quote or empty>", "reason": "<short>"}]}"""


# ---------------------------------------------------------------- deterministic

def inline_ids(answer):
    return set(re.findall(r"\[([A-Za-z0-9_]+_c\d{5})\]", answer or ""))


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def unsupported_entities(answer, texts):
    """Numbers and capitalised mid-sentence words in the answer found in no passage."""
    if not answer:
        return []
    body = re.sub(r"\[[A-Za-z0-9_]+_c\d{5}\]", " ", answer)
    corpus = " " + norm(" ".join(texts)) + " "
    found = set(re.findall(r"\b\d{3,4}\b", body))
    # capitalised words not at sentence start
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{3,})\b", body):
        found.add(m.group(1))
    return sorted(w for w in found if f" {norm(w).strip()} " not in corpus)


def gold_in_context(rec):
    gold = set(rec.get("gold_chunk_ids") or [])
    if not gold:
        return None
    got = gold & set(rec["retrieved_ids"])
    return len(got) == len(gold) if rec.get("gold_mode") == "all" else bool(got)


def deterministic(rec):
    ans = rec.get("answer") or ""
    cited = set(rec.get("citations") or []) | inline_ids(ans)
    retrieved = set(rec["retrieved_ids"])
    fabricated = sorted(cited - retrieved)
    answered = rec.get("status") == "answer"
    return {
        "behaviour_ok": rec.get("status") == rec["expected_behaviour"],
        "fabricated_citations": fabricated,
        "uncited_answer": answered and not cited,
        "gold_in_ctx": gold_in_context(rec),
        "unsupported_entities": unsupported_entities(ans, rec["retrieved_texts"].values())
                                if answered else [],
    }


# ---------------------------------------------------------------- judge

def _squash(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def quote_found(quote, passages_text):
    """True if the judge's quote is (nearly) verbatim in the passages:
    exact after normalisation, or one contiguous match covering >=80% of it."""
    q, p = _squash(quote), passages_text
    if len(q) < 8:
        return False
    if q in p:
        return True
    m = difflib.SequenceMatcher(None, q, p, autojunk=False).find_longest_match(0, len(q), 0, len(p))
    return m.size >= 0.8 * len(q)


def judge(client, model, rec):
    passages = "\n\n".join(f"[{i}]\n{t}" for i, t in rec["retrieved_texts"].items())
    user = f"PASSAGES\n\n{passages}\n\nANSWER\n\n{rec['answer']}"
    text, ti, to = call_claude(client, model, JUDGE_SYSTEM, user, temperature=0.0, max_tokens=3000)
    parsed = parse_json(text)
    if not parsed or not isinstance(parsed.get("claims"), list) or not parsed["claims"]:
        return None
    flat = _squash(" ".join(rec["retrieved_texts"].values()))
    claims = []
    for c in parsed["claims"]:
        if not isinstance(c, dict):
            continue
        if c.get("verdict") == "supported" and not quote_found(c.get("evidence", ""), flat):
            c["verdict"] = "unverified"
            c["reason"] = "evidence quote not found in passages; " + str(c.get("reason", ""))
        claims.append(c)
    if not claims:
        return None
    n_sup = sum(1 for c in claims if c.get("verdict") == "supported")
    return {"claims": claims, "n_claims": len(claims), "n_supported": n_sup,
            "n_unverified": sum(1 for c in claims if c.get("verdict") == "unverified"),
            "groundedness": n_sup / len(claims), "tokens": [ti, to]}


# ---------------------------------------------------------------- reporting

def pct(num, den):
    return f"{num}/{den}" if den else "-"


def report(rows):
    print(f"\n{'tier':<8}{'n':>3}  {'behav':>7} {'cite_ok':>8} {'gold_ctx':>9} "
          f"{'grounded':>9} {'flag_ent':>9}")
    groups = defaultdict(list)
    for r in rows:
        groups[r["tier"]].append(r)
    tiers = [t for t in TIER_ORDER if t in groups] + [t for t in groups if t not in TIER_ORDER]
    for t in tiers + ["ALL"]:
        rs = rows if t == "ALL" else groups[t]
        beh = sum(r["det"]["behaviour_ok"] for r in rs)
        cite_ok = sum(not r["det"]["fabricated_citations"] for r in rs)
        g = [r["det"]["gold_in_ctx"] for r in rs if r["det"]["gold_in_ctx"] is not None]
        gr = [r["judge"]["groundedness"] for r in rs if r.get("judge")]  # primary judge
        ent = sum(bool(r["det"]["unsupported_entities"]) for r in rs if r["status"] == "answer")
        n_ans = sum(r["status"] == "answer" for r in rs)
        gtxt = f"{sum(gr) / len(gr):.2f}" if gr else "-"
        print(f"{t:<8}{len(rs):>3}  {pct(beh, len(rs)):>7} {pct(cite_ok, len(rs)):>8} "
              f"{pct(sum(g), len(g)):>9} {gtxt:>9} {pct(ent, n_ans):>9}")

    # abstention / clarification, split by expected behaviour
    print("\nBehaviour confusion (rows = expected, cols = got):")
    kinds = ["answer", "abstain", "clarify", None]
    print(f"{'':<10}" + "".join(f"{str(k):>9}" for k in kinds))
    for e in ["answer", "abstain", "clarify"]:
        rs = [r for r in rows if r["expected_behaviour"] == e]
        if rs:
            print(f"{e:<10}" + "".join(f"{sum(r['status'] == k for r in rs):>9}" for k in kinds))
    false_abs = [r["id"] for r in rows if r["expected_behaviour"] == "answer" and r["status"] == "abstain"]
    missed = [r["id"] for r in rows if r["expected_behaviour"] == "abstain" and r["status"] != "abstain"]
    print(f"\nFalse abstentions on answerable questions: {false_abs or 'none'}")
    print(f"Failed to abstain (hallucination risk):    {missed or 'none'}")

    # model vs retrieval failure on answerable questions
    ans_rows = [r for r in rows if r["expected_behaviour"] == "answer"
                and r["det"]["gold_in_ctx"] is not None]
    no_ctx = [r for r in ans_rows if not r["det"]["gold_in_ctx"]]
    ctx_but_abstain = [r["id"] for r in ans_rows if r["det"]["gold_in_ctx"] and r["status"] != "answer"]
    print(f"\nAnswerable questions where retrieval missed gold: {len(no_ctx)}/{len(ans_rows)} "
          f"({[r['id'] for r in no_ctx]})")
    print(f"Gold was in context but model did not answer:     {ctx_but_abstain or 'none'}")

    pf = [r["id"] for r in rows if not r.get("parse_ok", True)]
    print(f"Parse failures / loose parses (harness, not model, problems): {pf or 'none'}")

    # behaviour conditional on retrieval: the number that isolates the model
    print("\nOn answerable questions, model behaviour by retrieval outcome:")
    for label, flag in (("gold in context", True), ("gold NOT in context", False)):
        rs = [r for r in rows if r["expected_behaviour"] == "answer"
              and r["det"]["gold_in_ctx"] is flag]
        c = defaultdict(int)
        for r in rs:
            c[str(r["status"])] += 1
        print(f"  {label:<20} n={len(rs):<3} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))

    fab = [(r["id"], r["det"]["fabricated_citations"]) for r in rows if r["det"]["fabricated_citations"]]
    print(f"Fabricated citations: {fab or 'none'}")
    bad = [(r["id"], [c for c in r["judge"]["claims"] if c.get("verdict") != "supported"])
           for r in rows if r.get("judge") and r["judge"]["groundedness"] < 1]
    if bad:
        print("\nAnswers with unsupported / unverified claims (primary judge):")
        for i, cs in bad:
            for c in cs:
                print(f"  {i}: [{c.get('verdict')}] {c.get('claim')}  -- {c.get('reason')}")


def judge_comparison(rows, judges):
    """Measure self-preference: same answers, independent judge vs self-judge."""
    if len(judges) < 2:
        return {}
    gen = next((r.get("model") for r in rows if r.get("model")), None)
    print("\nJudge comparison (same answers):")
    print(f"  {'judge':<22}{'role':<14}{'n':>4} {'grounded':>9} {'fully_ok':>9} "
          f"{'unsupp':>7} {'unverif':>8}")
    summary = {}
    for m in judges:
        js = [r["judges"][m] for r in rows if r.get("judges", {}).get(m)]
        if not js:
            continue
        role = "self-judge" if m == gen else "independent"
        g = sum(j["groundedness"] for j in js) / len(js)
        full = sum(j["groundedness"] == 1 for j in js)
        uns = sum(sum(c.get("verdict") == "unsupported" for c in j["claims"]) for j in js)
        unv = sum(j.get("n_unverified", 0) for j in js)
        summary[m] = {"role": role, "n": len(js), "groundedness": round(g, 3),
                      "fully_supported": full, "unsupported_claims": uns, "unverified_claims": unv}
        print(f"  {m:<22}{role:<14}{len(js):>4} {g:>9.2f} {full:>4}/{len(js):<4} {uns:>7} {unv:>8}")
    a, b = judges[0], judges[1]
    both = [r for r in rows if r.get("judges", {}).get(a) and r["judges"].get(b)]
    if both:
        agree = [r for r in both if (r["judges"][a]["groundedness"] == 1) == (r["judges"][b]["groundedness"] == 1)]
        dis = [r["id"] for r in both if r not in agree]
        print(f"  Answer-level agreement (fully supported or not): {len(agree)}/{len(both)}"
              f"   disagreements: {dis or 'none'}")
        summary["agreement"] = {"agree": len(agree), "n": len(both), "disagreements": dis}
        if a in summary and b in summary:
            gap = summary[b]["groundedness"] - summary[a]["groundedness"]
            print(f"  Self-preference gap ({b} minus {a}): {gap:+.3f}")
            summary["self_preference_gap"] = round(gap, 3)
    return summary


def write_spot_check(rows, n, seed, path="spot_check.md"):
    pool = [r for r in rows if r["status"] == "answer"]
    random.Random(seed).shuffle(pool)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Manual spot-check ({min(n, len(pool))} answers, seed {seed})\n\n"
                "For each: is every claim in the answer supported by the passages? "
                "Fill in `Reviewer verdict` (all supported / some unsupported), whether you "
                "agree with each judge, and who reviewed it.\n\n")
        for r in pool[:n]:
            f.write(f"---\n\n## {r['id']} ({r['tier']})\n\n**Q:** {r['question']}\n\n"
                    f"**Answer:** {r['answer']}\n\n")
            for m, j in (r.get("judges") or {}).items():
                if not j:
                    continue
                f.write(f"**Judge {m}:** {j['n_supported']}/{j['n_claims']} supported\n\n")
                for c in j["claims"]:
                    ev = f' — "{c.get("evidence")}"' if c.get("evidence") else ""
                    f.write(f"- [{c.get('verdict')}] {c.get('claim')}{ev}\n")
                f.write("\n")
            f.write("**Reviewer verdict:** \n\n**Agree with judges:** \n\n**Reviewed by:** \n\n"
                    "<details><summary>Passages</summary>\n\n")
            for i, t in r["retrieved_texts"].items():
                f.write(f"**{i}**\n\n{t}\n\n")
            f.write("</details>\n\n")
    print(f"Spot-check file -> {path}")


DEFAULT_JUDGES = ["claude-opus-5-5", DEFAULT_MODEL]  # primary (independent) first


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="generation_results.json")
    ap.add_argument("--out", default="generation_scores.json")
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES),
                    help="comma-separated; the FIRST is the primary (headline) judge")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--spot", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="spot-check sampling seed")
    args = ap.parse_args()
    judges = [m.strip() for m in args.judges.split(",") if m.strip()]

    with open(args.results, encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        print("No results to score.")
        sys.exit(1)

    prior = {}
    try:
        with open(args.out, encoding="utf-8") as f:
            prior = {r["id"]: r for r in json.load(f)}
    except (OSError, ValueError):
        pass

    client = None if args.no_judge else make_client()
    for r in rows:
        r["det"] = deterministic(r)
        r["judges"] = {}
        if args.no_judge or r.get("status") != "answer" or not r.get("answer"):
            continue
        old = prior.get(r["id"]) or {}
        for m in judges:
            cached = (old.get("judges") or {}).get(m)
            if cached and old.get("answer") == r["answer"]:
                r["judges"][m] = cached
                continue
            try:
                r["judges"][m] = judge(client, m, r)
                if r["judges"][m] is None:
                    print(f"  {r['id']}: {m} judge output unparseable")
            except Exception as e:
                print(f"  {r['id']}: {m} judge API error: {e}")
                r["judges"][m] = None
        r["judge"] = r["judges"].get(judges[0])  # primary judge drives the table
        r["judge_model"] = judges[0]

    if len(rows) <= 5:  # small test run: show per-question detail
        for r in rows:
            print(f"\n--- {r['id']} expected={r['expected_behaviour']} got={r['status']}")
            print("  deterministic:", json.dumps(r["det"]))
            for m, j in r.get("judges", {}).items():
                for c in (j or {}).get("claims", []):
                    print(f"  {m} [{c.get('verdict')}] {c.get('claim')}")
    if not args.no_judge:
        print(f"\nPrimary judge: {judges[0]}   (grounded column below)")
    report(rows)
    summary = judge_comparison(rows, judges) if not args.no_judge else {}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if summary:
        with open("judge_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("Judge summary -> judge_summary.json")
    print(f"\nScores -> {args.out}")
    write_spot_check(rows, args.spot, args.seed)


if __name__ == "__main__":
    main()