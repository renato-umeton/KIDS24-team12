# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains work from KIDS24 Biohackathon Team 12, focusing on improving Q&A capabilities for the Illumina DRAGEN Bio-IT Platform through two approaches:

1. **Fine-tuning approach**: Using LLaMA-Factory to fine-tune Meta-Llama-3.1-70B-Instruct on DRAGEN manual content
2. **RAG approach**: Implementing Retrieval Augmented Generation using both LangChain (with Ollama) and OpenAI

The project processes DRAGEN manual PDFs to create training datasets and enable accurate question-answering about DRAGEN command-line usage.

## Repository Structure

```
vm_files/Jose/
├── files/
│   ├── 4-2_manual.pdf              # Source DRAGEN manual
│   ├── dataset.json                # Fine-tuning dataset in Alpaca format
│   ├── metadata.json               # Per-page prompt generation metadata
│   └── RAG_raw_data.json          # Processed PDF data for RAG
├── LLaMA-Factory/                  # Git submodule (fork from hiyouga/LLaMA-Factory)
├── FT-1-dataset_generator.ipynb   # Generates fine-tuning dataset from PDF
├── FT-2-fine_tuning.ipynb         # Fine-tunes LLaMA using LLaMA-Factory
├── RAG-langchain_approach.ipynb   # RAG implementation with LangChain + Ollama
└── RAG-openai.ipynb               # RAG implementation with OpenAI API
```

## Key Components

### Fine-tuning Pipeline

The fine-tuning approach uses a two-stage process:

1. **Dataset Generation** (`FT-1-dataset_generator.ipynb`):
   - Processes PDF manual page-by-page using PyPDF2
   - Uses OpenAI GPT-4o assistants API with file_search to generate prompts
   - Creates 10-20 Alpaca-format prompts per page
   - Stores results in `files/dataset.json` and `files/metadata.json`
   - Handles preceding page context for continuity

2. **Model Fine-tuning** (`FT-2-fine_tuning.ipynb`):
   - Uses LLaMA-Factory framework for fine-tuning
   - Base model: `meta-llama/Meta-Llama-3.1-70B-Instruct`
   - Fine-tuning method: LoRA adapters
   - Training configs stored in `LLaMA-Factory/saves/Llama-3.1-70B-Instruct/lora/`
   - Supports 4-bit quantization for inference

### RAG Pipeline

Two RAG implementations are provided:

1. **LangChain + Ollama** (`RAG-langchain_approach.ipynb`):
   - Uses LangChain document loaders and text splitters
   - Embeddings: `sentence-transformers/all-mpnet-base-v2`
   - Vector store: FAISS
   - LLM: Ollama with llama3.1 model
   - Requires local Ollama installation

2. **OpenAI** (`RAG-openai.ipynb`):
   - Uses GPT-4o for image analysis of PDF pages
   - Embeddings: `text-embedding-3-small`
   - Cosine similarity for content retrieval
   - Response generation: GPT-4o

### Alpaca Dataset Format

The fine-tuning dataset follows Alpaca format with these fields:
- `instruction`: The task or query (required)
- `input`: Context or details (optional, empty string if unused)
- `output`: Expected response (required)
- `system`: System instruction (always "Do not add information not explicitly stated or speculate.")
- `history`: Array of prior [instruction, response] pairs (optional)

## Development Environment

This project was developed on a high-performance computing cluster with:
- 8x NVIDIA A100-SXM4-80GB GPUs
- 1507 GB RAM
- CUDA 12.1
- Python 3.10 (conda environment: py310)

### Key Dependencies

**Fine-tuning:**
- `torch==2.3.1` with CUDA support
- `llamafactory` (LLaMA-Factory package)
- `openai==1.57.0`
- `PyPDF2`
- `huggingface_hub`

**RAG (LangChain):**
- `langchain` and `langchain-community`
- `sentence-transformers`
- `faiss-gpu`
- `pypdf`
- `langchain_ollama`

**RAG (OpenAI):**
- `openai`
- `pdf2image`
- `pdfminer.six`
- `scikit-learn`
- `pandas`

## Working with the Codebase

### Fine-tuning Workflow

**Generate dataset from PDF:**
```python
# In FT-1-dataset_generator.ipynb
# Set OpenAI API key in oai_token.txt
# Run cells to process PDF and generate dataset.json
```

**Fine-tune model:**
```bash
# Terminal: Launch LLaMA-Factory Gradio interface
cd /path/to/LLaMA-Factory
llamafactory-cli webui

# Access via SSH tunnel if on remote server
ssh -L 7860:localhost:7860 <server>
```

**Load fine-tuned model:**
```python
# In FT-2-fine_tuning.ipynb
from llamafactory.chat import ChatModel

args = dict(
    model_name_or_path="meta-llama/Meta-Llama-3.1-70B-Instruct",
    adapter_name_or_path="saves/Llama-3.1-70B-Instruct/lora/16_batch_size",
    template="llama3",
    finetuning_type="lora",
    quantization_bit=4,
)
chat_model = ChatModel(args)
```

### RAG Workflow

**LangChain approach:**
```python
# Requires Ollama running locally
# Terminal: ollama serve & ollama pull llama3.1

# In RAG-langchain_approach.ipynb
# Load PDF, create embeddings, query with RetrievalQA
```

**OpenAI approach:**
```python
# Set OpenAI API key in oai_token.txt
# In RAG-openai.ipynb
# Process PDF to embeddings CSV, then query
```

## Important Notes

### API Keys and Tokens
- OpenAI API key should be stored in `vm_files/Jose/oai_token.txt`
- HuggingFace token should be stored in `vm_files/Jose/hf_token.txt`
- These files are NOT tracked in git (.gitignore)

### Git Submodules
The repository includes a forked version of LLaMA-Factory as a submodule. To initialize:
```bash
git submodule update --init --recursive
```

### Dataset Update Workflow
When regenerating the dataset:
```bash
# Copy updated dataset to LLaMA-Factory
cp vm_files/Jose/files/dataset.json vm_files/Jose/LLaMA-Factory/data/dragen_alpaca.json
```

### Training Data Characteristics
- Source: DRAGEN v4.2 manual (740 pages)
- Dataset contains 6000+ prompts in Alpaca format
- Focused on command-line syntax and platform usage
- System prompt emphasizes factual accuracy without speculation

## Model Details

**Fine-tuned Model:**
- Base: Meta-Llama-3.1-70B-Instruct
- Method: LoRA fine-tuning
- Template: llama3
- Quantization: 4-bit for inference
- Training variations tested with batch sizes of 8 and 16

**RAG Models:**
- LangChain: llama3.1 via Ollama
- OpenAI: GPT-4o for both analysis and response generation
