#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script
"""
from tqdm import tqdm
import json
import os
from datetime import datetime
import time
from utils import *
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim

from Model import Bert_model

if conf.pretrained_model == "bert":
    from transformers import BertTokenizer, BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)
elif conf.pretrained_model == "roberta":
    from transformers import RobertaTokenizer, RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)

# create output paths
if conf.mode == "train":
    model_dir_name = conf.model_save_name + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=False)
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')
else:
    saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, 'inference_only_' + model_dir_name)
    results_path = os.path.join(model_dir, "results")
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')

op_list = read_txt(conf.op_list_file, log_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)

train_data, train_examples, op_list, const_list = read_examples(
    input_path=conf.train_file, tokenizer=tokenizer, op_list=op_list, const_list=const_list, log_file=log_file
)
valid_data, valid_examples, op_list, const_list = read_examples(
    input_path=conf.valid_file, tokenizer=tokenizer, op_list=op_list, const_list=const_list, log_file=log_file
)
test_data, test_examples, op_list, const_list = read_examples(
    input_path=conf.test_file, tokenizer=tokenizer, op_list=op_list, const_list=const_list, log_file=log_file
)

kwargs = {"examples": train_examples[:10], "tokenizer": tokenizer, "option": conf.option, "is_training": True,
          "max_seq_length": conf.max_seq_length}
train_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = valid_examples[:10]; kwargs["is_training"] = False
valid_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = test_examples[:10]
test_features = convert_examples_to_features(**kwargs)

def train():
    model = Bert_model(hidden_size=model_config.hidden_size, dropout_rate=conf.dropout_rate)
    model = nn.DataParallel(model)
    model.to(conf.device)
    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()

    train_iterator = DataLoader(is_training=True, data=train_features,
                                batch_size=conf.batch_size, shuffle=True)

    k = 0; record_k = 0; record_loss = 0.0; start_time = time.time()
    print(">>> Training for", conf.epoch, "epochs")
    print(">>> Saving will trigger every", conf.report, "steps")
    print(">>> Loss will be reported every", conf.report_loss, "steps")
    print(">>> Saving path root:", saved_model_path)

    for epoch in range(conf.epoch):
        print(f"\n>>> Starting epoch {epoch+1}/{conf.epoch}")
        train_iterator.reset()
        for x in train_iterator:
            k += 1
            input_ids = torch.tensor(x['input_ids']).to(conf.device)
            input_mask = torch.tensor(x['input_mask']).to(conf.device)
            segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
            label = torch.tensor(x['label']).to(conf.device)

            model.zero_grad(); optimizer.zero_grad()
            logits = model(True, input_ids, input_mask, segment_ids, device=conf.device)
            this_loss = criterion(logits.view(-1, logits.shape[-1]), label.view(-1)).sum()
            record_loss += this_loss.item() * 100; record_k += 1
            this_loss.backward(); optimizer.step()

            if k > 1 and k % conf.report_loss == 0:
                avg_loss = record_loss / record_k if record_k > 0 else 0.0
                print(f"[Step {k}] Reporting loss: {avg_loss:.3f}")
                write_log(log_file, "%d : loss = %.3f" % (k, avg_loss))
                record_loss = 0.0; record_k = 0

            if k > 1 and k % conf.report == 0:
                print(f"\n>>> [Step {k}] Saving checkpoint")
                model.eval()
                cost_time = time.time() - start_time
                print(f"    Time since last report: {cost_time:.3f}s")
                write_log(log_file, "%d : time = %.3f " % (k // conf.report, cost_time))
                start_time = time.time()
                save_dir = os.path.join(saved_model_path, 'loads', str(k // conf.report))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                print("    Model saved successfully at", save_dir)
                try:
                    results_dir = os.path.join(results_path, 'loads', str(k // conf.report))
                    os.makedirs(results_dir, exist_ok=True)
                    evaluate(valid_examples, valid_features, model, results_dir, 'valid')
                except KeyError as e:
                    print("    Skipping validation due to KeyError:", e)
                model.train()

def evaluate(data_ori, data, model, ksave_dir, mode='valid'):
    pred_list, pred_unk = [], []
    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)
    data_iterator = DataLoader(is_training=False, data=data,
                               batch_size=conf.batch_size_test, shuffle=False)
    all_logits, all_filename_id, all_ind = [], [], []
    with torch.no_grad():
        for x in tqdm(data_iterator):
            input_ids, input_mask, segment_ids = x['input_ids'], x['input_mask'], x['segment_ids']
            filename_id, ind = x["filename_id"], x["ind"]
            # pad batch if smaller
            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids]:
                if ori_len < conf.batch_size_test:
                    each_item += [[0]*len(each_item[0])] * (conf.batch_size_test - ori_len)
            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)
            logits = model(True, input_ids, input_mask, segment_ids, device=conf.device)
            all_logits.extend(logits.tolist())
            all_filename_id.extend(filename_id)
            all_ind.extend(ind)
    output_prediction_file = os.path.join(ksave_dir_mode, "predictions.json")
    if mode == "valid":
        print_res = retrieve_evaluate(all_logits, all_filename_id, all_ind,
                                      output_prediction_file, conf.valid_file, topn=conf.topn)
    else:
        print_res = retrieve_evaluate(all_logits, all_filename_id, all_ind,
                                      output_prediction_file, conf.test_file, topn=conf.topn)
    write_log(log_file, print_res); print(print_res)

if __name__ == '__main__':
    train()
