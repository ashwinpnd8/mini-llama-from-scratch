# Mini Llama From Scratch

A compact educational language-model project built with PyTorch.

## Included features

- Byte Pair Encoding tokenizer written from scratch
- Tokenizer training, saving, loading, encoding, and decoding
- Llama-style RMSNorm
- Grouped-Query Attention
- Rotary Positional Embeddings
- Optional QK normalization
- Causal attention masking
- SwiGLU feed-forward network
- Residual connections
- Weight-tied language-model head
- Dataset and DataLoader pipeline
- AdamW training loop
- Gradient clipping
- Checkpoint saving
- Top-k text generation
- CPU and CUDA support
- Command-line arguments

## Project structure

```text
mini_llama_upgraded/
├── config.py
├── dataset.py
├── generate.py
├── model.py
├── README.md
├── requirements.txt
├── sample_corpus.txt
├── smoke_test.py
├── tokenizer.py
├── train.py
└── utils.py
```

## Setup on Windows PowerShell

Open the project folder:

```powershell
cd path\to\mini_llama_upgraded
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the smoke test:

```powershell
python smoke_test.py
```

## Train

Quick training with the included corpus:

```powershell
python train.py --corpus sample_corpus.txt --epochs 30
```

The trained files are saved inside the `artifacts` folder:

```text
artifacts/
├── config.json
├── model.pt
└── tokenizer.json
```

## Generate text

```powershell
python generate.py --prompt "This document" --max-new-tokens 30
```

Example with different sampling settings:

```powershell
python generate.py --prompt "Attention" --temperature 0.7 --top-k 10
```

## Use a larger dataset

Create a UTF-8 text file with one training example or paragraph per line:

```powershell
python train.py --corpus your_dataset.txt --epochs 50
```

A larger and cleaner corpus will produce better output. This is still a small educational model and is not intended to match production LLMs.


**Mini Llama-style Language Model from Scratch**

Built a modular causal language model in PyTorch featuring a custom BPE tokenizer, Grouped-Query Attention, Rotary Positional Embeddings, RMSNorm, SwiGLU feed-forward layers, residual connections, causal masking, checkpointing, and top-k text generation. Implemented an end-to-end training and inference pipeline with CPU/GPU support.

## Skills demonstrated

Python, PyTorch, NLP, transformers, tokenization, deep learning, model training, command-line interfaces, software modularity, and Git/GitHub.
