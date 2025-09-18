import argparse
import json
import os
import sys

from finqa_utils import table_row_to_text

sys.path.insert(0, '../utils/')
from general_utils import table_row_to_text


def convert_test(json_in, json_out, topn, max_len):
    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text
        table = each_data["table"]

        all_retrieved = each_data["table_retrieved"] + each_data["text_retrieved"]
        sorted_dict = sorted(all_retrieved, key=lambda kv: kv["score"], reverse=True)

        acc_len = 0
        all_text_in, all_table_in = {}, {}

        for tmp in sorted_dict:
            if len(all_table_in) + len(all_text_in) >= topn:
                break
            this_sent_ind = int(tmp["ind"].split("_")[1])
            this_sent = table_row_to_text(table[0], table[this_sent_ind]) if "table" in tmp["ind"] else all_text[this_sent_ind]

            if acc_len + len(this_sent.split(" ")) < max_len:
                if "table" in tmp["ind"]:
                    all_table_in[tmp["ind"]] = this_sent
                else:
                    all_text_in[tmp["ind"]] = this_sent
                acc_len += len(this_sent.split(" "))
            else:
                break

        this_model_input = []
        sorted_dict_table = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) < len(pre_text):
                this_model_input.append(tmp)
        this_model_input.extend(sorted_dict_table)
        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) >= len(pre_text):
                this_model_input.append(tmp)

        each_data["annotation"]["model_input"] = this_model_input

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[convert_test] Wrote {len(data)} examples to {json_out}")


def convert_train(json_in, json_out, topn, max_len):
    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text
        gold_inds = each_data["annotation"]["gold_ind"]
        table = each_data["table"]

        all_retrieved = each_data["table_retrieved"] + each_data["text_retrieved"]
        false_retrieved = [tmp for tmp in all_retrieved if tmp["ind"] not in gold_inds]
        sorted_dict = sorted(false_retrieved, key=lambda kv: kv["score"], reverse=True)

        acc_len = 0
        all_text_in, all_table_in = {}, {}

        for tmp in gold_inds:
            if "table" in tmp:
                all_table_in[tmp] = gold_inds[tmp]
            else:
                all_text_in[tmp] = gold_inds[tmp]

        context = " ".join(gold_inds.values())
        acc_len = len(context.split(" "))

        for tmp in sorted_dict:
            if len(all_table_in) + len(all_text_in) >= topn:
                break
            this_sent_ind = int(tmp["ind"].split("_")[1])
            this_sent = table_row_to_text(table[0], table[this_sent_ind]) if "table" in tmp["ind"] else all_text[this_sent_ind]

            if acc_len + len(this_sent.split(" ")) < max_len:
                if "table" in tmp["ind"]:
                    all_table_in[tmp["ind"]] = this_sent
                else:
                    all_text_in[tmp["ind"]] = this_sent
                acc_len += len(this_sent.split(" "))
            else:
                break

        this_model_input = []
        sorted_dict_table = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) < len(pre_text):
                this_model_input.append(tmp)
        this_model_input.extend(sorted_dict_table)
        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) >= len(pre_text):
                this_model_input.append(tmp)

        each_data["annotation"]["model_input"] = this_model_input

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[convert_train] Wrote {len(data)} examples to {json_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert retriever outputs into generator inputs.")
    parser.add_argument("--retriever_file", type=str, required=True, help="Retriever output JSON file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save converted JSON")
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True, help="Dataset split")
    parser.add_argument("--topn", type=int, default=3, help="Max number of retrieved evidence to keep")
    parser.add_argument("--max_len", type=int, default=290, help="Max sequence length (words)")

    args = parser.parse_args()

    if args.split == "train":
        convert_train(args.retriever_file, args.save_path, args.topn, args.max_len)
    else:
        convert_test(args.retriever_file, args.save_path, args.topn, args.max_len)
