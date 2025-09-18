import os
import json
import argparse
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from difflib import SequenceMatcher

try:
    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
except ImportError:
    rouge = None

smoothie = SmoothingFunction().method4

def edit_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def token_f1(gold_tokens, pred_tokens):
    gold_set, pred_set = set(gold_tokens), set(pred_tokens)
    tp = len(gold_set & pred_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gold_set) if gold_set else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def load_json(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return []
    with open(path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--sample", type=int, default=3)
    parser.add_argument("--export", default="analysis.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    # Load results
    full_path = os.path.join(args.results_dir, "full_results.json")
    nbest_path = os.path.join(args.results_dir, "nbest_predictions.json")
    preds_path = os.path.join(args.results_dir, "predictions.json")

    full_results = load_json(full_path)
    nbest = load_json(nbest_path)
    preds = load_json(preds_path)

    print(f"[INFO] Loaded: full={len(full_results)} nbest={len(nbest)} preds_keys={list(preds.keys()) if isinstance(preds, dict) else type(preds)}")

    mismatches, bleu_scores, rouge_scores, edit_scores, f1_scores = [], [], [], [], []

    for i, rec in enumerate(full_results):
        gold_prog = rec.get("gold_prog", [])
        pred_prog = rec.get("pred_prog", [])
        gold_ans  = rec.get("gold_ans")
        pred_ans  = rec.get("pred_ans")

        if i < 5:
            print(f"[DEBUG Example {i}] keys={list(rec.keys())}")
            print(f"  gold_prog: {gold_prog}")
            print(f"  pred_prog: {pred_prog}")
            print(f"  gold_ans: {gold_ans}")
            print(f"  pred_ans: {pred_ans}")
            if not gold_prog and not pred_prog:
                print(f"  [WARNING] Empty programs, item snippet: {json.dumps(rec, indent=2)[:200]}...")

        # detect mismatches
        if gold_prog != pred_prog or gold_ans != pred_ans:
            mismatches.append({
                "id": rec.get("id", str(i)),
                "gold_prog": gold_prog,
                "pred_prog": pred_prog,
                "gold_ans": gold_ans,
                "pred_ans": pred_ans,
            })

        # metrics
        gp, pp = list(map(str, gold_prog)), list(map(str, pred_prog))
        if gp and pp:
            try:
                bleu_scores.append(sentence_bleu([gp], pp, smoothing_function=smoothie))
            except Exception as e:
                print(f"[WARN] BLEU failed on id={rec.get('id')} ({e})")
                bleu_scores.append(0.0)

            if rouge:
                try:
                    rouge_scores.append(rouge.score(" ".join(gp), " ".join(pp))["rougeL"].fmeasure)
                except Exception as e:
                    print(f"[WARN] ROUGE failed on id={rec.get('id')} ({e})")
                    rouge_scores.append(0.0)
            else:
                rouge_scores.append(0.0)

            try:
                edit_scores.append(edit_sim(gp, pp))
            except Exception as e:
                print(f"[WARN] Edit sim failed ({e})")
                edit_scores.append(0.0)

            try:
                f1_scores.append(token_f1(gp, pp))
            except Exception as e:
                print(f"[WARN] Token F1 failed ({e})")
                f1_scores.append(0.0)

    # KPIs
    total = len(full_results)
    mismatch_count = len(mismatches)
    err_count = sum(1 for r in full_results if not r.get("pred_prog"))

    print("\n==== MISMATCH ANALYSIS ====")
    print(f"Total examples: {total}")
    print(f"Mismatches: {mismatch_count} ({mismatch_count/total:.1%})")
    print(f"Errors (exec failures): {err_count} ({err_count/total:.1%})")
    print(f"Program Accuracy: {(1 - mismatch_count/total):.3f}")
    print(f"Execution Accuracy: {(1 - err_count/total):.3f}")
    print(f"Avg BLEU: {sum(bleu_scores)/len(bleu_scores):.3f}" if bleu_scores else "Avg BLEU: N/A")
    print(f"Avg ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.3f}" if rouge_scores else "Avg ROUGE-L: N/A")
    print(f"Avg Edit Similarity: {sum(edit_scores)/len(edit_scores):.3f}" if edit_scores else "Avg Edit Sim: N/A")
    print(f"Avg Token F1: {sum(f1_scores)/len(f1_scores):.3f}" if f1_scores else "Avg Token F1: N/A")

    # Show mismatches
    print(f"\n==== SAMPLE MISMATCHES (showing {args.sample}) ====")
    for rec in mismatches[:args.sample]:
        print(f"\n--- ID: {rec['id']} ---")
        print(f"Gold program : {rec['gold_prog'] or 'EMPTY'}")
        print(f"Pred program : {rec['pred_prog'] or 'EMPTY'}")
        print(f"Gold answer  : {rec['gold_ans']}")
        print(f"Pred answer  : {rec['pred_ans']}")

    # Export Excel
    out_path = os.path.join(args.results_dir, args.export)
    df_full = pd.DataFrame(full_results)
    df_mismatches = pd.DataFrame(mismatches)
    kpis = {
        "Total": total,
        "Mismatches": mismatch_count,
        "Errors": err_count,
        "Program Accuracy": (1 - mismatch_count/total),
        "Execution Accuracy": (1 - err_count/total),
        "Avg BLEU": sum(bleu_scores)/len(bleu_scores) if bleu_scores else None,
        "Avg ROUGE-L": sum(rouge_scores)/len(rouge_scores) if rouge_scores else None,
        "Avg Edit Similarity": sum(edit_scores)/len(edit_scores) if edit_scores else None,
        "Avg Token F1": sum(f1_scores)/len(f1_scores) if f1_scores else None,
    }
    df_kpis = pd.DataFrame([kpis])

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_full.to_excel(writer, sheet_name="FullResults", index=False)
        df_mismatches.to_excel(writer, sheet_name="Mismatches", index=False)
        df_kpis.to_excel(writer, sheet_name="KPIs", index=False)

    print(f"\n[INFO] Exported Excel analysis to {out_path}")

if __name__ == "__main__":
    main()
