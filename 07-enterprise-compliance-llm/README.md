# Enterprise Document Intelligence & Compliance Assistant (Milestone 1)

A RAG (Retrieval-Augmented Generation) system that ingests documents and answers questions with source citations. Built as the foundation for an enterprise compliance checking system.

## 🎯 Current Features (Milestone 1)

- 📄 **Document Ingestion**: Parse PDF and DOCX files
- ✂️ **Smart Chunking**: Semantic text splitting with configurable overlap
- 🔍 **Semantic Search**: Embed documents and find relevant content using vector similarity
- 💬 **RAG Q&A**: Ask questions and get answers grounded in your documents
- 📌 **Source Citations**: Every answer includes document references
- 🏠 **Model**: Uses Groq (llama-3.1-8b-instant) for LLM inference

## 🏗️ Architecture
PDF/DOCX → Parser → Chunker → Embedder → Qdrant Vector DB
↓
User Question → Embed → Search → Retrieve → LLM → Answer + Citations


## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Groq (llama-3.1-8b-instant) | Best inference, zero cost |
| **Embeddings** | all-MiniLM-L6-v2 | Sentence embeddings (384-dim) |
| **Vector DB** | Qdrant | Semantic search |
| **Backend** | FastAPI | REST API |
| **Document Parsing** | PyPDF2, python-docx | Extract text |

## 🚀 Quick Start

### Prerequisites

# Install Groq
# Install Docker Desktop for Qdrant

Setup
# 1. Clone and navigate to project
cd 07-enterprise-compliance-llm

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Qdrant (vector database)
docker-compose up -d

# 5. Start the API server
python serving/api.py

# In a new terminal
python test_rag.py
Or use the API directly:

# Ingest a document
curl -X POST http://localhost:8001/ingest \
  -F "file=@sample_contract.pdf"

# Ask a question
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the liability cap?"}'


📁 Project Structure
07-enterprise-compliance-llm/
├── ingestion/
│   ├── __init__.py
│   ├── document_parser.py    # PDF/DOCX text extraction
│   ├── chunker.py            # Semantic text splitting
│   └── embed_and_store.py    # Vector embedding + Qdrant storage
├── rag/
│   ├── __init__.py
│   ├── prompts.py            # Centralized prompt templates
│   ├── retriever.py          # Vector similarity search
│   └── synthesizer.py        # LLM answer generation
├── serving/
│   ├── __init__.py
│   └── api.py                # FastAPI server
├── docker-compose.yml        # Qdrant service
├── .env                      # Configuration (LLM provider, etc.)
├── test_rag.py               # End-to-end test
├── requirements.txt
└── README.md


🧪 Testing
# Start the API server first
python serving/api.py

# Run tests in another terminal
python test_rag.py

# Expected output:
=== Testing RAG System ===

✓ API is running
✓ Document ingested

Q: What is the liability cap?
A: The liability cap is $500,000... [Source: contract.pdf]
✓ Citations included

🎓 Concepts Demonstrated
Concept	Implementation
RAG	Document retrieval + LLM synthesis
Prompt Engineering	System prompts for controlled generation
Embeddings	Semantic search with sentence-transformers
Chunking	Overlapping window strategy
Vector Search	Cosine similarity in Qdrant
API Design	RESTful endpoints with FastAPI
Inference	Deployment with Groq

## 🎯 Milestone 2: Fine-Tuned Compliance Classifier

### What's New
- **Fine-tuned Model**: Custom LoRA adapter trained on 200 legal clauses
- **Violation Detection**: Automatically identifies GDPR and SOX violations
- **Risk Scoring**: Classifies severity as HIGH/MEDIUM/LOW
- **Audit Endpoint**: Upload contract → Get violation report

### Architecture Addition
Fine-Tuned Llama-3.2-1B (LoRA)
↓
Contract Clauses → Violation Detection → Risk Report
↓
GDPR Art.6, SOX Sec.404, etc.

### New Concepts Learned
- **QLoRA Fine-Tuning**: 4-bit quantization enables CPU training
- **Synthetic Data Generation**: Using LLM to create training data
- **Instruction Tuning**: Formatting data for task-specific training
- **Model Evaluation**: Accuracy metrics for multi-class classification

### New Endpoints
# Audit a contract for violations
curl -X POST http://localhost:8001/audit \
  -F "file=@contract.pdf"

# Files Added
fine_tuning/
├── generate_dataset.py    # Creates 200 synthetic clauses
├── train_lora.py          # QLoRA fine-tuning on CPU
└── eval_model.py          # Accuracy evaluation

agents/tools/
└── compliance_tool.py     # Fine-tuned model inference

|__ test_audit.py

## How to Run
# 1. Generate training data (uses Groq - free)
python fine_tuning/generate_dataset.py

# 2. Train the model (30-60 min on CPU)
python fine_tuning/train_lora.py

# 3. Test the audit endpoint
python test_audit.py
---

🚀 Execution Order
# 1. Generate dataset (5-10 minutes)
python fine_tuning/generate_dataset.py

# 2. Train model (30-60 minutes on CPU)
python fine_tuning/train_lora.py

# 3. Evaluate model
python fine_tuning/eval_model.py

# 4. Restart API and test
python serving/api.py
python test_audit.py

📜 License
MIT