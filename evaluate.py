"""Score the NER models against a hand-labelled answer key.

Inputs (CSV files, see data/ for examples):
  samples.csv        columns: sample_id,text
  gold_entities.csv  columns: sample_id,entity,type      (type is PERSON, ORG or LOC)

Usage:
  python evaluate.py
  python evaluate.py --models spacy,transformer --errors-out results/errors.csv

An entity counts as correct only if the text AND the type both match (case ignored).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

from ner_engine import ENTITY_TYPES, MODELS, NERError, extract


def norm(s):
    """Lower-case, collapse spaces, and ignore a leading 'the' ('the Reserve Bank' == 'Reserve Bank')."""
    s = " ".join(s.lower().split()).rstrip(".")   # 'Acme Robotics Inc.' == 'Acme Robotics Inc'
    return s[4:] if s.startswith("the ") else s


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def ratio(a, b):
    return a / b if b else 0.0


def evaluate(samples, gold_rows, model):
    gold = defaultdict(set)
    for r in gold_rows:
        gold[r["sample_id"].strip()].add((norm(r["entity"]), r["type"].strip().upper()))

    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in ENTITY_TYPES}
    errors = []
    for s in samples:
        sid = s["sample_id"].strip()
        if sid not in gold:
            continue
        pred = {(norm(e["text"]), e["label"]) for e in extract(s["text"], model)}
        for ent, typ in pred & gold[sid]:
            counts[typ]["tp"] += 1
        for ent, typ in pred - gold[sid]:
            counts[typ]["fp"] += 1
            errors.append({"sample_id": sid, "model": model, "error": "false_positive", "entity": ent, "type": typ})
        for ent, typ in gold[sid] - pred:
            if typ in counts:
                counts[typ]["fn"] += 1
            errors.append({"sample_id": sid, "model": model, "error": "false_negative", "entity": ent, "type": typ})
    return counts, errors


def report(model, counts):
    print(f"\n=== {MODELS[model]} ===")
    print(f"{'Type':<8}{'TP':>5}{'FP':>5}{'FN':>5}{'Precision':>11}{'Recall':>9}{'F1':>7}")
    total = {"tp": 0, "fp": 0, "fn": 0}
    for typ in ENTITY_TYPES + ["ALL"]:
        c = total if typ == "ALL" else counts[typ]
        if typ != "ALL":
            for k in total:
                total[k] += c[k]
        p = ratio(c["tp"], c["tp"] + c["fp"])
        r = ratio(c["tp"], c["tp"] + c["fn"])
        f1 = ratio(2 * p * r, p + r)
        print(f"{typ:<8}{c['tp']:>5}{c['fp']:>5}{c['fn']:>5}{p:>11.2f}{r:>9.2f}{f1:>7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="data/samples.csv")
    ap.add_argument("--gold", default="data/gold_entities.csv")
    ap.add_argument("--models", default="spacy,transformer", help="comma separated: spacy,transformer")
    ap.add_argument("--errors-out", default="results/errors.csv")
    args = ap.parse_args()

    samples, gold_rows = read_csv(args.samples), read_csv(args.gold)
    scored = {r["sample_id"].strip() for r in gold_rows}
    print(f"Scoring {len(scored)} of {len(samples)} paragraphs (only those with an answer key): "
          + ", ".join(sorted(scored, key=lambda s: int(s.lstrip('P') or 0) if s.lstrip('P').isdigit() else 0)))
    all_errors = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            counts, errors = evaluate(samples, gold_rows, model)
        except NERError as err:
            print(f"\n[{model}] skipped: {err}", file=sys.stderr)
            continue
        report(model, counts)
        all_errors += errors

    if all_errors:
        os.makedirs(os.path.dirname(args.errors_out) or ".", exist_ok=True)
        with open(args.errors_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "model", "error", "entity", "type"])
            w.writeheader()
            w.writerows(all_errors)
        print(f"\nMistakes written to {args.errors_out}")


if __name__ == "__main__":
    main()
