#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for retriever
"""
from tqdm import tqdm
import os
from datetime import datetime
import argparse
from utils import *
from config import parameters as conf
from torch import nn
import torch

from Model import Bert_model

# ------------------------
# Parse CLI arguments
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None,
                    help="Path to trained model checkpoint (.pt)")
parser.add_argument("--test_file", type=str, default=None,
                    help="JSON file to run inference on (dev/test)")
parser.add_argument("--save_path", type=str, default=None,
                    help="Output path for predictions")
args = parser.parse_args()

# Resolve paths: CLI > config
model_path = args.model_path if args.model_path else conf.saved_model_path
test_file = args.test_file if args.test_file else conf.test_file
save_path = args.save_path if args.save_path else "predictions.json"

print(">>> Using model_path:", model_path)
print(">>> Using test_file:", test_file)
print(">>> Saving outputs to:", save_path)

# ------------------------
# Setup tokenizer + model config
# ------------------------
if conf.pretrained_model == "bert":
    from transformers import BertTokenizer, BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)
elif conf.pretrained_model == "roberta":
    from transformers import RobertaTokenizer, RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)

# ------------------------
# Setup dirs
# ------------------------
model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + conf.model_save_name
model_dir = os.path.join(conf.output_path, 'inference_only_' + model_dir_name)
results_path = os.path.join(model_dir, "results")
os.makedirs(results_path, exist_ok=False)
log_file = os.path.join(results_path, 'log.txt')

# ------------------------
# Prepare examples/features
# ------------------------
op_list = read_txt(conf.op_list_file, log_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]

test_data, test_examples, op_list, const_list = read_examples(
    input_path=test_file, tokenizer=tokenizer,
    op_list=op_list, const_list=const_list, log_file=log_file
)

kwargs = {"examples": test_examples,
          "tokenizer": tokenizer,
          "option": conf.option,
          "is_training": False,
          "max_seq_length": conf.max_seq_length}
test_features = convert_examples_to_features(**kwargs)

# ------------------------
# Generate
# ------------------------
def generate(data_ori, data, model, ksave_dir, mode='test'):
    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(is_training=False, data=data,
                               batch_size=conf.batch_size_test, shuffle=False)

    all_logits, all_filename_id, all_ind = [], [], []
    with torch.no_grad():
        for x in tqdm(data_iterator):
            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            filename_id = x["filename_id"]
            ind = x["ind"]

            # pad batch if smaller than batch_size_test
            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids]:
                if ori_len < conf.batch_size_test:
                    pad_x = [0] * len(each_item[0])
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)

            logits = model(True, input_ids, input_mask,
                           segment_ids, device=conf.device)

            all_logits.extend(logits.tolist())
            all_filename_id.extend(filename_id)
            all_ind.extend(ind)

    output_prediction_file = save_path
    print(">>> Writing predictions to:", output_prediction_file)
    print_res = retrieve_evaluate(all_logits, all_filename_id, all_ind,
                                  output_prediction_file, test_file, topn=conf.topn)

    write_log(log_file, print_res)
    print(print_res)


def generate_test():
    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate)
    model = nn.DataParallel(model)
    model.to(conf.device)

    print(">>> Loading weights from:", model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    generate(test_data, test_features, model, results_path, mode='test')


if __name__ == '__main__':
    generate_test()
