import os
import torch

class parameters():

    prog_name = "generator"

    # ---------- paths: prefer env, else safe local defaults ----------
    try:
        root_path = os.environ["FINQA_ROOT"]
        print(f"[config] FINQA_ROOT found: {root_path}")
    except KeyError:
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        print(f"[config] FINQA_ROOT not set, using default: {root_path}")

    try:
        output_path = os.environ["FINQA_OUTPUT"]
        print(f"[config] FINQA_OUTPUT found: {output_path}")
    except KeyError:
        output_path = os.path.join(root_path, "generator_ckpt")
        print(f"[config] FINQA_OUTPUT not set, using default: {output_path}")

    try:
        cache_dir = os.environ["HF_HOME"]
        print(f"[config] HF_HOME found: {cache_dir}")
    except KeyError:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        print(f"[config] HF_HOME not set, using default: {cache_dir}")

    # files from the retriever results
    try:
        train_file = os.environ["FINQA_TRAIN"]
        print(f"[config] FINQA_TRAIN found: {train_file}")
    except KeyError:
        train_file = "./dev_retrieve.json"
        print(f"[config] FINQA_TRAIN not set, using default: {train_file}")

    try:
        valid_file = os.environ["FINQA_VALID"]
        print(f"[config] FINQA_VALID found: {valid_file}")
    except KeyError:
        valid_file = "./dev_retrieve.json"
        print(f"[config] FINQA_VALID not set, using default: {valid_file}")

    try:
        test_file = os.environ["FINQA_TEST"]
        print(f"[config] FINQA_TEST found: {test_file}")
    except KeyError:
        test_file = "./dev_retrieve.json"
        print(f"[config] FINQA_TEST not set, using default: {test_file}")

    op_list_file = "operation_list.txt"
    const_list_file = "constant_list.txt"

    # ---------- model ----------
    pretrained_model = "bert"            # bert | roberta | finbert | longformer
    model_size = "bert-base-uncased"    # change via env/CLI if desired
    model_save_name = "generator-bert-base-try"

    # retrieval/program modes
    retrieve_mode = "single"             # single | slide | gold | none
    program_mode = "seq"                 # seq | nest

    # runtime
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[config] Using device: {device}")
    mode = "train"

    # path used only for test/inference mode (can be overridden)
    try:
        saved_model_path = os.environ["FINQA_GEN_CKPT"]
        print(f"[config] FINQA_GEN_CKPT found: {saved_model_path}")
    except KeyError:
        saved_model_path = os.path.join(output_path, "model.pt")
        print(f"[config] FINQA_GEN_CKPT not set, using default: {saved_model_path}")

    
    build_summary = False

    # architecture/training hyperparams
    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512   # 2k for longformer, 512 for others
    max_program_length = 30
    n_best_size = 20
    dropout_rate = 0.1

    # reduced for small GPU
    batch_size = 4
    batch_size_test = 4
    epoch = 300
    learning_rate = 1e-5

    report = 300
    report_loss = 100

    max_step_ind = 11
