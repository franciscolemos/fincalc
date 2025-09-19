#!/usr/bin/env bash
# Quick reference commands for the ConvFinQA retriever and generator workflows.
#
# Running this script prints out the recommended commands with repository-aware
# paths so they can be copied and executed manually. The dynamic paths ensure
# the instructions work regardless of the local checkout directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
fi
RETRIEVER_DIR="${PROJECT_ROOT}/code/finqanet_retriever"
GENERATOR_DIR="${PROJECT_ROOT}/code/finqanet_generator"
DATA_DIR="${PROJECT_ROOT}/data"

cat <<EOF_INSTRUCTIONS
Train the retriever model

cd "${RETRIEVER_DIR}"

python Main.py \\
  --train_file "${DATA_DIR}/train_turn.json" \
  --valid_file "${DATA_DIR}/dev_turn.json" \
  --output_dir "${RETRIEVER_DIR}/output"


output/retriever-roberta-base_20250914193244/saved_model

python Test.py \\
  --model_path output/retriever-roberta-base_20250914193244/saved_model/loads/2/model.pt \
  --test_file "${DATA_DIR}/dev_turn.json" \
  --save_path retriever_outputs.json
#       Replace <timestamp> with the actual folder name you saw.
#       This will run the retriever on the dev set and write predictions (retriever_outputs.json).

cd "${GENERATOR_DIR}"

python Convert.py \\
  --retriever_file "${RETRIEVER_DIR}/retriever_outputs.json" \
  --save_path "${GENERATOR_DIR}/dev_retrieve.json" \
  --split dev


python Main.py \\
  --train_file "${GENERATOR_DIR}/dev_retrieve.json" \
  --valid_file "${GENERATOR_DIR}/dev_retrieve.json" \
  --test_file  "${GENERATOR_DIR}/dev_retrieve.json" \
  --output_dir "${GENERATOR_DIR}/generator_ckpt" \
  --mode train


export FINQA_GEN_CKPT="${GENERATOR_DIR}/generator_ckpt/generator-bert-base-try_20250914225507/saved_model/loads/21/model.pt"

cd "${GENERATOR_DIR}"

python Convert.py \\
  --retriever_file "${RETRIEVER_DIR}/retriever_outputs.json" \
  --save_path "${GENERATOR_DIR}/test_retrieve.json" \
  --split test

export FINQA_GEN_CKPT="${GENERATOR_DIR}/generator_ckpt/generator-bert-base-try_20250914225507/saved_model/loads/21/model.pt"

python Main.py \\
  --test_file "${GENERATOR_DIR}/test_retrieve.json" \
  --output_dir "${GENERATOR_DIR}/generator_ckpt" \
  --mode test

python parse_results.py --sample 2 --head 1
python parse_results.py --sample 2 --head 3


python analyze_results.py \\
  --results_dir "${GENERATOR_DIR}/generator_ckpt/generator-bert-base-try_20250914225507/results/loads/21/valid" \
  --sample 2

python analyze_mismatches.py \\
  --results_dir "${GENERATOR_DIR}/generator_ckpt/generator-bert-base-try_20250914225507/results/loads/21/valid" \
  --sample 3

python analyze_mismatches.py \\
  --results_dir "${GENERATOR_DIR}/generator_ckpt/generator-bert-base-try_20250914225507/results/loads/21/valid" \
  --export analysis.xlsx \
  --sample 5
EOF_INSTRUCTIONS
