Train the retriever model


cd /home/francis/tomoro/ConvFinQA/code/finqanet_retriever

python Main.py \
  --train_file ../../data/train_turn.json \
  --valid_file ../../data/dev_turn.json \
  --output_dir ./output


output/retriever-roberta-base_20250914193244/saved_model

python Test.py \
  --model_path output/retriever-roberta-base_20250914193244/saved_model/loads/2/model.pt \
  --test_file ../../data/dev_turn.json \
  --save_path retriever_outputs.json
#	Replace <timestamp> with the actual folder name you saw.
#	This will run the retriever on the dev set and write predictions (retriever_outputs.json).

cd /home/francis/tomoro/ConvFinQA/code/finqanet_generator

python Convert.py \
  --retriever_file ../finqanet_retriever/retriever_outputs.json \
  --save_path ./dev_retrieve.json \
  --split dev


python Main.py \
  --train_file ./dev_retrieve.json \
  --valid_file ./dev_retrieve.json \
  --test_file  ./dev_retrieve.json \
  --output_dir ./generator_ckpt \
  --mode train


export FINQA_GEN_CKPT=./generator_ckpt/generator-bert-base-try_20250914225507/saved_model/loads/21/model.pt

cd /home/francis/tomoro/ConvFinQA/code/finqanet_generator

python Convert.py \
  --retriever_file ../finqanet_retriever/retriever_outputs.json \
  --save_path ./test_retrieve.json \
  --split test

export FINQA_GEN_CKPT=./generator_ckpt/generator-bert-base-try_20250914225507/saved_model/loads/21/model.pt

python Main.py \
  --test_file ./test_retrieve.json \
  --output_dir ./generator_ckpt \
  --mode test

python parse_results.py --sample 2 --head 1
python parse_results.py --sample 2 --head 3


python analyze_results.py \
  --results_dir ./generator_ckpt/generator-bert-base-try_20250914225507/results/loads/21/valid \
  --sample 2

python analyze_mismatches.py \
  --results_dir ./generator_ckpt/generator-bert-base-try_20250914225507/results/loads/21/valid \
  --sample 3

python analyze_mismatches.py \
  --results_dir ./generator_ckpt/generator-bert-base-try_20250914225507/results/loads/21/valid \
  --export analysis.xlsx \
  --sample 5




