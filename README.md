## Mistral_7b RAG
#### - Zahir Mahammad
---
Install using requirements.txt
```
pip install -r requirements.txt
```

In the script "_run_mistral.py_" adjust the folder path of pdf/text files and faiss store path

Run the script
```
python run_mistral.py
```
---



# RAG_Chatbot_Research_Project

Welcome to the RAG Chatbot Research Project repository! This project aims to create a RAG chatbot utilizing advanced Large Language Models (LLMs) and a Retrieval-Augmented Generation (RAG) system. The repository includes all relevant code, datasets, and tools required for this research.

---

## 📋 Project Overview

This study is designed to fine-tune LLMs for RAG chatbot applications without human participants, relying entirely on numerical performance metrics.

### Project Highlights
- Utilizes **GPUs** for accelerated processing.
- Recommended configuration: **15 GB RAM** and **15-20 GB of GPU shared memory** for optimal performance.

---

## 📊 Datasets

The project involves training LLMs on two primary datasets, with additional datasets for evaluation:

1. **Meadow-MedQA** (Hugging Face): Includes 10,178 training entries, each with inputs, instructions, and outputs.
2. **MedMCQA** (Kaggle): A multiple-choice dataset with 194,000 questions spanning 21 RAG subjects. Data is split into training, testing, and validation sets.

**Evaluation Datasets:**
- **USMLE MedQA** for MCQ evaluation.
- **Comprehensive RAG Q&A** for subjective question assessment.

#### Dataset References
| Datasets                     | Related Research Paper                                                                 | Website Link                                                                                     |
|------------------------------|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| RAG-Meadow-MedQA          | Medalpaca - an open-source collection of RAG conversational AI models and training data (Han. T., 2023) | [RAG-Meadow-MedQA](https://huggingface.co/datasets/medalpaca/RAG_meadow_medqa)           |
| MedMCQA                       | MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset for RAG domain Question Answering (Pal, 2022)  | [MedMCQA](https://www.kaggle.com/datasets/thedevastator/medmcqa-RAG-mcq-dataset)             |
| USMLE MedQA                   | Towards Expert-Level RAG Question Answering with Large Language Models (Singhal. K, 2023)   | [USMLE MedQA](https://www.kaggle.com/datasets/moaaztameer/medqa-usmle)                           |
| Comprehensive RAG Q&A     | A question-entailment approach to question answering (Abacha. A.B, 2019)                     | [Comprehensive RAG Q&A](https://www.kaggle.com/datasets/thedevast

## 🗄️ Vector Database Architecture

The chatbot’s knowledge base was built from RAG texts, parsed and chunked into segments. 

### Core Elements:
- **Document Parsing**: Split text and PDFs into 500-character chunks with a 50-character overlap.
- **Embeddings**: Generated using the `gtr-t5-large` model and stored in a FAISS vector database.

Files:
- `index.faiss`: Stores vector data.
- `index.pkl`: Metadata and configuration.

| **Parameter**              | **Value**     |
|----------------------------|---------------|
| Chunk Size                 | 500 characters|
| Overlap Size               | 50 characters |
| Embedding Model            | gtr-t5-large  |
| Computation Time           | 7,560 seconds |
| `index.faiss` Size         | 753 MB        |
| `index.pkl` Size           | 122 MB        |

---

## 🔧 Fine-Tuning Large Language Models (LLMs)

The project leverages three distinct LLMs to power the chatbot:

1. **Flan-T5-Large**: A quantized encoder-decoder model trained with LoRA for memory efficiency.
2. **LLaMA-2-7B**: A decoder-only, chat-optimized model with 4-bit quantization.
3. **Mistral-7B**: A GPTQ-quantized model similar to LLaMA but with optimized efficiency.

---

## 🔄 Chatbot System Design

### 1. Base Model with RAG
Combines user query embeddings with the `FAISS` vector database for context-aware generation.

| Parameter      | Value    |
|----------------|----------|
| `do_sample`    | True     |
| `top_k`        | 1        |
| `temperature`  | 0.1      |
| `max_new_tokens` | 150   |

### 2. Fine-tuned Model without RAG
Directly generates responses using the LLM without additional context from a vector database.

### 3. Fine-tuned Model with RAG
Combines the fine-tuned model and RAG, pulling relevant RAG information from the vector database before generating responses.

---

## 🌐 Web-Based Chatbot Interface

The project includes a web-based chatbot using **Flask** for the backend, enabling real-time interactions. The frontend, designed with HTML, CSS, and JavaScript, provides intuitive messaging functions.

---

## 📖 Citation and Resources

- **Meadow-MedQA** dataset on Hugging Face
- **MedMCQA** dataset on Kaggle

Additional libraries and tools:
- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Flask](https://flask.palletsprojects.com/)

--- 

Thank you for exploring the RAG Chatbot Research Project!
