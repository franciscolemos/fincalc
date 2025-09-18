#!/usr/bin/env python3
import os, json, argparse
import pandas as pd

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="./generator_ckpt/generator-bert-base-try_20250914225507/results/loads/21/valid",
                        help="Directory containing predictions.json, nbest_predictions.json, full_results.json, etc.")
    parser.add_argument("--sample", type=int, default=2, help="How many sample items to show")
    parser.add_argument("--head", type=int, default=2, help="How many head rows from full results to show")
    args = parser.parse_args()

    preds = load_json(os.path.join(args.results_dir, "predictions.json")) or {}
    nbest = load_json(os.path.join(args.results_dir, "nbest_predictions.json")) or {}
    full = None
    try:
        full = pd.read_json(os.path.join(args.results_dir, "full_results.json"))
    except Exception:
        full = pd.DataFrame()
    errors = None
    try:
        errors = pd.read_json(os.path.join(args.results_dir, "full_results_error.json"))
    except Exception:
        errors = pd.DataFrame()

    print("==== SUMMARY KPIs ====")
    print(f"Predictions: {len(preds) if isinstance(preds, dict) else 0} items")
    print(f"N-best predictions: {len(nbest) if isinstance(nbest, dict) else 0} items")
    print(f"Full results: {len(full)} items")
    print(f"Errors: {len(errors)} items\n")

    print("==== SAMPLES (truncated) ====\n")

    # Predictions sample
    if isinstance(preds, dict) and preds:
        print("--- Predictions (sample) ---")
        for i, (k,v) in enumerate(preds.items()):
            if i >= args.sample: break
            print(f"ID: {k}, Predicted program: {str(v)[:120]}...")
        print()

    # N-best sample
    if isinstance(nbest, dict) and nbest:
        print("--- N-Best Predictions (sample) ---")
        for i, (k,v) in enumerate(nbest.items()):
            if i >= args.head: break
            print(f"ID: {k}")
            if isinstance(v, list):
                for cand in v[:args.sample]:
                    print(f" Candidate: {str(cand)[:120]}...")
        print()

    # Full results head
    if not full.empty:
        print("--- Full Results (head) ---")
        print(full.head(args.head))
        print()

    # Errors head
    if not errors.empty:
        print("--- Full Results Errors (head) ---")
        print(errors.head(args.head))
        print()
