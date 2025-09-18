# ConvFinQA

ConvFinQA is a pipeline for conversational financial question answering.  
It consists of two main components:  

1. **Retriever** — identifies relevant evidence from financial reports.  
2. **Generator** — takes the retrieved evidence and generates reasoning steps and answers.  

---

## 📂 Project Structure
```
code/
├── finqanet_retriever/   # Retriever model training and inference
│   ├── Main.py           # Train retriever
│   ├── Test.py           # Evaluate retriever / produce retriever_outputs.json
│   └── config.py
│
├── finqanet_generator/   # Generator model training and inference
│   ├── Convert.py        # Convert retriever outputs → generator inputs
│   ├── Main.py           # Train generator
│   ├── Test.py           # Evaluate generator
│   └── config.py
│
data/                     # Input datasets (train/dev/test JSON)
```

---

## 🚀 Pipeline Overview

### 1. Train the Retriever
```bash
cd code/finqanet_retriever

python Main.py \
  --train_file ../../data/train_turn.json \
  --valid_file ../../data/dev_turn.json \
  --output_dir ./output
```
This trains the retriever model (RoBERTa-based).  
A checkpoint is saved at \`output/.../model.pt\`.

---

### 2. Evaluate the Retriever
```bash
python Test.py \
  --model_path output/retriever-roberta-base_TIMESTAMP/saved_model/loads/2/model.pt \
  --test_file ../../data/dev_turn.json \
  --save_path retriever_outputs.json
```
- Loads the trained retriever checkpoint  
- Runs inference on the dev/test set  
- Produces **\`retriever_outputs.json\`** with predicted supporting facts  

---

### 3. Convert Retriever Outputs → Generator Inputs
```bash
cd ../finqanet_generator

python Convert.py \
  --retriever_file ../finqanet_retriever/retriever_outputs.json \
  --save_path ./dev_retrieve.json \
  --split dev
```
This produces a structured dataset (\`dev_retrieve.json\`) suitable for the generator.

---

### 4. Train the Generator
```bash
python Main.py \
  --train_file ./dev_retrieve.json \
  --valid_file ./dev_retrieve.json \
  --test_file ./dev_retrieve.json \
  --output_dir ./generator_ckpt \
  --mode train
```

---

### 5. Evaluate the Generator
```bash
# Convert test set
python Convert.py \
  --retriever_file ../finqanet_retriever/retriever_outputs.json \
  --save_path ./test_retrieve.json \
  --split test

# Run evaluation
python Main.py \
  --test_file ./test_retrieve.json \
  --output_dir ./generator_ckpt \
  --mode test
```

---

## 📦 Outputs
- **Retriever**: \`retriever_outputs.json\` → top-k evidence predictions  
- **Generator**: predicted answers and reasoning programs  

---

## ⚡ Notes
- \`.pt\` files = PyTorch checkpoints (weights only).  
- Large files like datasets (\`train_turn.json\`, \`dev_turn.json\`) are not included in the repo — place them under \`data/\`.  
- Add \`*.zip\`, \`data/\`, and \`output/\` to \`.gitignore\` to keep the repo lightweight.  
