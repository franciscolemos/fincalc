import os, json, argparse, pandas as pd
from utils import eval_program

def load_json(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--sample", type=int, default=2)
    args = parser.parse_args()

    preds = load_json(os.path.join(args.results_dir, "predictions.json")) or {}
    nbest = load_json(os.path.join(args.results_dir, "nbest_predictions.json")) or {}
    full = load_json(os.path.join(args.results_dir, "full_results.json")) or []
    errors = load_json(os.path.join(args.results_dir, "full_results_error.json")) or []

    print("==== DETAILED GOLD vs PRED ANALYSIS ====\n")

    count = 0
    for idx, cand_list in nbest.items():
        if count >= args.sample: break
        example = cand_list[0]
        print(f"--- Example ID: {idx} ---")
        print(f"[Source ID] {example.get('id')}")

        # Input text (truncate for readability)
        question = " ".join(example.get("question_tokens", [])[:30])
        print(f"INPUT TEXT: {question} ...")

        # Programs
        pred_prog = example.get("pred_prog", [])
        gold_prog = example.get("ref_prog", [])

        # Evaluate answers
        try:
            pred_ans = eval_program(pred_prog, example.get("numbers"))
        except Exception:
            pred_ans = "?"
        try:
            gold_ans = eval_program(gold_prog, example.get("numbers"))
        except Exception:
            gold_ans = "?"

        print("MODEL PREDICTION:")
        print(f"   Program: {pred_prog}")
        print(f"   ⇒ Predicted answer: {pred_ans}")
        print("GOLD REFERENCE:")
        print(f"   Program: {gold_prog}")
        print(f"   ⇒ Gold answer: {gold_ans}")

        print("WORKFLOW TRACE:")
        print(f"   INPUT TEXT --> MODEL PROGRAM --> {pred_ans}")
        print(f"             --> GOLD PROGRAM  --> {gold_ans}")
        print("   " + ("✅ MATCH" if str(pred_ans)==str(gold_ans) else "❌ MISMATCH"))
        print()
        count += 1

    print("==== SUMMARY KPIs ====")
    print(f"Predictions: {len(preds)} items")
    print(f"N-best predictions: {len(nbest)} items")
    print(f"Full results: {len(full)} items")
    print(f"Errors: {len(errors)} items")

if __name__ == "__main__":
    main()
