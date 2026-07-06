# Enterprise Document Intelligence & Compliance Assistant

An AI system that understands legal documents and automatically detects compliance violations.

## 🎯 What It Does

### 1. Document Q&A (RAG System)
Ask questions about any uploaded document and get answers with source citations.

### 2. Compliance Checking (Fine-tuned Model)
Automatically identifies GDPR and SOX violations in contract clauses.

## 🏗️ Architecture
┌─────────────────────────┐
│ FastAPI Server │
│ (serving/api.py) │
└──────────┬──────────────┘
│
┌───────────────────┼───────────────────┐
↓ ↓ ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ /ingest │ │ /query │ │ /audit │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
↓ ↓ ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Parse → Chunk│ │Retrieve → LLM│ │Parse → Model │
│ → Embed → Qdrant│ │ → Answer │ │ → Violations│
└──────────────┘ └──────────────┘ └──────────────┘


## 📁 Project Structure
├── ingestion/ # Document processing
│ ├── document_parser.py # PDF/DOCX → text
│ ├── chunker.py # Text → semantic chunks
│ └── embed_and_store.py # Chunks → vectors → Qdrant
│
├── rag/ # Question answering
│ ├── prompts.py # LLM prompt templates
│ ├── retriever.py # Vector similarity search
│ └── synthesizer.py # Generate answers via Groq
│
├── compliance.py # Violation detection
├── compliance_lora/ # Fine-tuned model weights
│
├── fine_tuning/ # Model training
│ ├── generate_dataset.py # Create training data
│ ├── train_lora.py # Training script
│ └── dataset/
│ └── compliance_data.json # 200 labeled clauses
│
├── serving/
│ └── api.py # FastAPI endpoints
│
├── tests/
│ ├── test_rag.py # RAG system tests
│ └── test_audit.py # Audit system tests
│
├── docker-compose.yml # Qdrant service



## 🚀 Quick Start

# 1. Start Qdrant
docker-compose up -d

# 2. Start API
python serving/api.py

# 3. Test RAG
python tests/test_rag.py

# 4. Test Compliance
python tests/test_audit.py

🔧 Tech Stack
# Component	Technology
LLM Inference	Groq (Llama-3.1-8B)
Fine-tuned Model	Llama-3.2-1B + LoRA
Vector Database	Qdrant
Embeddings	all-MiniLM-L6-v2
Backend	FastAPI
Training	Google Colab (T4 GPU)

🎓 Concepts Demonstrated
# Milestone 1: RAG Pipeline
Document parsing & chunking strategies

Vector embeddings & semantic search

Prompt engineering for accurate citations

Retrieval-Augmented Generation

# Milestone 2: Fine-Tuned Classifier
Synthetic data generation for training

QLoRA fine-tuning (4-bit quantization)

LoRA adapters for efficient model updates

Multi-class compliance classification

📊 Training Details
Model: Llama-3.2-1B (1.2B parameters)

Method: QLoRA (only 0.07% parameters trained)

Dataset: 200 synthetic legal clauses

Platform: Google Colab free T4 GPU

Training Time: ~3 minutes

Model Size: ~63 MB

🧪 Testing
# Test RAG pipeline
python tests/test_rag.py

# Test compliance checker
python compliance.py

# Test full audit
python tests/test_audit.py
📝 API Endpoints
Endpoint	Method	Description
/ingest	POST	Upload document
/query	POST	Ask question
/audit	POST	Check compliance
/health	GET	Health check

🔮 Future Enhancements
Agent orchestration for multi-step workflows

Support for more compliance frameworks (HIPAA, PCI-DSS)

Batch document processing

Web UI with Streamlit/Chainlit

---