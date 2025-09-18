class parameters():

    prog_name = "retriever"

    # set up your own path here
    root_path = "/home/francis/tomoro/ConvFinQA/"
    output_path = "output/"
    cache_dir = "cache/"

    # model choice: bert, roberta
    pretrained_model = "roberta"
    model_size = "roberta-base" # "distilroberta-base"   # or "roberta-large"

    # build save name automatically from model_size
    model_save_name = f"retriever-{model_size}"

    # use "train_turn.json", "dev_turn.json", and "test_turn.json"
    train_file = root_path + "data/train_turn.json"
    valid_file = root_path + "data/dev_turn.json"
    test_file = root_path + "data/test_turn_private.json"

    op_list_file = "operation_list.txt"
    const_list_file = "constant_list.txt"

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    mode = "train"
    resume_model_path = ""

    # to load the trained model in test time
    saved_model_path = output_path + "model.pt"
    build_summary = False

    option = "rand"
    neg_rate = 3
    topn = 5

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512
    max_program_length = 100
    n_best_size = 20
    dropout_rate = 0.1

    batch_size = 16
    batch_size_test = 16
    epoch = 20
    learning_rate = 2e-5

    report = 20   # save model every 20 steps
    report_loss = 5
