#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for generator
"""
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import time
from utils import *
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim
from Model_new import Bert_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Evaluate generator")
    parser.add_argument("--train_file", type=str, default=None, help="Training data (JSON)")
    parser.add_argument("--valid_file", type=str, default=None, help="Validation data (JSON)")
    parser.add_argument("--test_file",  type=str, default=None, help="Test data (JSON)")
    parser.add_argument("--output_dir", type=str, default=None, help="Root output directory")
    parser.add_argument("--mode", choices=["train","test"], default=None, help="Run mode")
    return parser.parse_args()


# ---- CLI overrides config.py ----
args = parse_args()
if args.train_file:  conf.train_file  = args.train_file
if args.valid_file:  conf.valid_file  = args.valid_file
if args.test_file:   conf.test_file   = args.test_file
if args.output_dir:  conf.output_path = args.output_dir
if args.mode:        conf.mode        = args.mode

# ---- Tokenizer / model config ----
if conf.pretrained_model == "bert":
    print("Using bert")
    from transformers import BertTokenizer, BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size, cache_dir=conf.cache_dir)
    model_config = BertConfig.from_pretrained(conf.model_size, cache_dir=conf.cache_dir)
elif conf.pretrained_model == "roberta":
    print("Using roberta")
    from transformers import RobertaTokenizer, RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size, cache_dir=conf.cache_dir)
    model_config = RobertaConfig.from_pretrained(conf.model_size, cache_dir=conf.cache_dir)
elif conf.pretrained_model == "finbert":
    print("Using finbert")
    from transformers import BertTokenizer, BertConfig
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=conf.cache_dir)
    model_config = BertConfig.from_pretrained(conf.model_size, cache_dir=conf.cache_dir)
elif conf.pretrained_model == "longformer":
    print("Using longformer")
    from transformers import LongformerTokenizer, LongformerConfig
    tokenizer = LongformerTokenizer.from_pretrained(conf.model_size, cache_dir=conf.cache_dir)
    model_config = LongformerConfig.from_pretrained(conf.model_size, cache_dir=conf.cache_dir)

# ---- Output dirs ----
if conf.mode == "train":
    model_dir_name = conf.model_save_name + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    log_file = os.path.join(results_path, 'log.txt')
else:
    # inference
    saved_model_path = conf.saved_model_path
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, 'inference_only_' + model_dir_name)
    results_path = os.path.join(model_dir, "results")
    os.makedirs(results_path, exist_ok=True)
    log_file = os.path.join(results_path, 'log.txt')

# ---- Operator / constant lists ----
op_list = read_txt(conf.op_list_file, log_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)
print(op_list); print(const_list)

# ---- Load datasets ----
train_data, train_examples, op_list, const_list = read_examples(
    input_path=conf.train_file, tokenizer=tokenizer,
    op_list=op_list, const_list=const_list, log_file=log_file)
valid_data, valid_examples, op_list, const_list = read_examples(
    input_path=conf.valid_file, tokenizer=tokenizer,
    op_list=op_list, const_list=const_list, log_file=log_file)
test_data, test_examples, op_list, const_list = read_examples(
    input_path=conf.test_file, tokenizer=tokenizer,
    op_list=op_list, const_list=const_list, log_file=log_file)

kwargs = {"examples": train_examples,
          "tokenizer": tokenizer,
          "max_seq_length": conf.max_seq_length,
          "max_program_length": conf.max_program_length,
          "is_training": True,
          "op_list": op_list, "op_list_size": len(op_list),
          "const_list": const_list, "const_list_size": len(const_list),
          "verbose": True}
train_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = valid_examples; kwargs["is_training"] = False
valid_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = test_examples
test_features = convert_examples_to_features(**kwargs)


# =========================
# evaluate() 
# =========================
def evaluate(data_ori, data, model, ksave_dir, mode='valid'):
    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(
        is_training=False, data=data,
        batch_size=conf.batch_size_test,
        reserved_token_size=reserved_token_size, shuffle=False)

    all_results = []
    with torch.no_grad():
        for x in tqdm(data_iterator):
            input_ids   = x['input_ids']
            input_mask  = x['input_mask']
            segment_ids = x['segment_ids']
            program_ids = x['program_ids']
            program_mask= x['program_mask']
            option_mask = x['option_mask']

            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids, program_ids, program_mask, option_mask]:
                if ori_len < conf.batch_size_test:
                    pad_x = [0] * len(each_item[0])
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)

            input_ids   = torch.tensor(input_ids).to(conf.device)
            input_mask  = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)
            program_ids = torch.tensor(program_ids).to(conf.device)
            program_mask= torch.tensor(program_mask).to(conf.device)
            option_mask = torch.tensor(option_mask).to(conf.device)

            logits = model(False, input_ids, input_mask, segment_ids,
                           option_mask, program_ids, program_mask, device=conf.device)

            for this_logit, this_id in zip(logits.tolist(), x["unique_id"]):
                all_results.append(RawResult(unique_id=int(this_id),
                                             logits=this_logit, loss=None))

    out_pred  = os.path.join(ksave_dir_mode, "predictions.json")
    out_nbest = os.path.join(ksave_dir_mode, "nbest_predictions.json")
    out_eval  = os.path.join(ksave_dir_mode, "full_results.json")
    out_err   = os.path.join(ksave_dir_mode, "full_results_error.json")

    all_predictions, all_nbest = compute_predictions(
        data_ori, data, all_results,
        n_best_size=conf.n_best_size,
        max_program_length=conf.max_program_length,
        tokenizer=tokenizer,
        op_list=op_list, op_list_size=len(op_list),
        const_list=const_list, const_list_size=len(const_list))
    write_predictions(all_predictions, out_pred)
    write_predictions(all_nbest, out_nbest)

    original_file = conf.valid_file if mode == "valid" else conf.test_file
    exe_acc, prog_acc = evaluate_result(out_nbest, original_file, out_eval, out_err,
                                        program_mode=conf.program_mode)
    res = f"exe acc: {exe_acc} prog acc: {prog_acc}"
    write_log(log_file, res); print(res)
    return exe_acc, prog_acc


# =========================
# Training
# =========================
def train():
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        write_log(log_file, attr + " = " + str(conf.__dict__[attr]))
    write_log(log_file, "#######################################################")

    model = Bert_model(num_decoder_layers=conf.num_decoder_layers,
                       hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,
                       program_length=conf.max_program_length,
                       input_length=conf.max_seq_length,
                       op_list=op_list, const_list=const_list)
    model = nn.DataParallel(model)
    model.to(conf.device)

    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()

    train_iterator = DataLoader(
        is_training=True, data=train_features, batch_size=conf.batch_size,
        reserved_token_size=reserved_token_size, shuffle=True)

    k = 0; record_k = 0; record_loss = 0.0; start_time = time.time()

    for _ in range(conf.epoch):
        train_iterator.reset()
        for x in train_iterator:
            k += 1
            input_ids   = torch.tensor(x['input_ids']).to(conf.device)
            input_mask  = torch.tensor(x['input_mask']).to(conf.device)
            segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
            program_ids = torch.tensor(x['program_ids']).to(conf.device)
            program_mask= torch.tensor(x['program_mask']).to(conf.device)
            option_mask = torch.tensor(x['option_mask']).to(conf.device)

            model.zero_grad(); optimizer.zero_grad()
            logits = model(True, input_ids, input_mask, segment_ids,
                           option_mask, program_ids, program_mask, device=conf.device)

            loss = criterion(logits.view(-1, logits.shape[-1]), program_ids.view(-1))
            loss = (loss * program_mask.view(-1)).sum() / program_mask.sum()

            record_loss += loss.item(); record_k += 1
            loss.backward(); optimizer.step()

            if k > 1 and k % conf.report_loss == 0:
                write_log(log_file, f"{k} : loss = {record_loss/record_k:.3f}")
                record_loss = 0.0; record_k = 0

            if k > 1 and k % conf.report == 0:
                print("Round:", k / conf.report)
                model.eval()
                cost_time = time.time() - start_time
                write_log(log_file, f"{k//conf.report} : time = {cost_time:.3f} ")
                start_time = time.time()

                save_dir = os.path.join(saved_model_path, 'loads', str(k // conf.report))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

                res_dir = os.path.join(results_path, 'loads', str(k // conf.report))
                os.makedirs(res_dir, exist_ok=True)
                try:
                    evaluate(valid_examples, valid_features, model, res_dir, 'valid')
                finally:
                    model.train()


# =========================
# Testing
# =========================
def test():
    print(f"[test] Loading model from {conf.saved_model_path}")
    model = Bert_model(num_decoder_layers=conf.num_decoder_layers,
                       hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,
                       program_length=conf.max_program_length,
                       input_length=conf.max_seq_length,
                       op_list=op_list, const_list=const_list)

    # load state dict and fix DataParallel "module." prefix if present
    state_dict = torch.load(conf.saved_model_path, map_location=conf.device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model = nn.DataParallel(model)
    model.to(conf.device)
    model.eval()

    evaluate(test_examples, test_features, model, results_path, 'test')



if __name__ == '__main__':
    if conf.mode == "train":
        train()
    elif conf.mode == "test":
        test()
