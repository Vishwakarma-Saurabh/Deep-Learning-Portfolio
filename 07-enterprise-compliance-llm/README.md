# Enterprise Document Intelligence & Compliance Assistant

An end-to-end AI system that understands legal documents, answers questions with citations, detects compliance violations, and executes multi-step workflows autonomously using LLM agents.

---

## 🎯 What It Does

### 1. Document Q&A (RAG)
Ask natural language questions about any uploaded document and get answers grounded in the source material with exact citations.

### 2. Compliance Violation Detection
Automatically identifies GDPR and SOX violations in contract clauses using a fine-tuned Llama model with severity scoring.

### 3. Autonomous AI Agent
Give a single complex command and the agent plans, executes, and reports - searching documents, auditing clauses, generating reports, and emailing results.

---

## 🏗️ Architecture
┌──────────────────────────────────────────────────────────────┐
│ FastAPI Server │
│ (serving/api.py) │
└────────────┬────────────┬──────────────┬──────────────────────┘
↓ ↓ ↓
┌──────────┐ ┌──────────┐ ┌──────────────────┐
│ /ingest │ │ /query │ │ /agent │
└────┬─────┘ └────┬─────┘ └────────┬─────────┘
↓ ↓ ↓
┌──────────────┐ ┌──────────┐ ┌─────────────────┐
│ Parse → Chunk│ │Retrieve │ │ Orchestrator │
│ → Embed → │ │→ Generate│ │ (ReAct Loop) │
│ Qdrant │ │→ Answer │ │ ↓ ↓ ↓ │
└──────────────┘ └──────────┘ │Search Audit │
│Report Email │
└─────────────────┘


### Data Flow
Document Upload → Parse Text → Semantic Chunks → Vector Embeddings → Qdrant
↓
User Question → Embed Query → Vector Search → Retrieve Chunks → LLM Answer

User Command → Agent Plans → Executes Tools → Multi-Step Result


---

## 📁 Project Structure
07-enterprise-compliance-llm/
│
├── data/ # Documents for testing
│ └── sample_contract.txt # 50+ clause test contract
│
├── ingestion/ # Document processing pipeline
│ ├── init.py
│ ├── document_parser.py # PDF/DOCX/TXT → clean text
│ ├── chunker.py # Text → semantic chunks (300 chars, 50 overlap)
│ └── embed_and_store.py # Chunks → vectors → Qdrant storage
│
├── rag/ # Retrieval-Augmented Generation
│ ├── init.py
│ ├── prompts.py # Centralized prompt templates
│ ├── retriever.py # Vector similarity search via Qdrant
│ └── synthesizer.py # Groq LLM answer generation
│
├── agents/ # Autonomous AI agent system
│ ├── init.py
│ ├── orchestrator.py # ReAct loop: Thought → Action → Observation
│ └── tools/ # Agent's toolbox
│ ├── init.py
│ ├── search_tool.py # Wraps RAG retriever
│ ├── audit_tool.py # Wraps compliance checker
│ ├── report_tool.py # Generates formatted reports via Groq
│ └── email_tool.py # Sends emails via SMTP (with simulation fallback)
│
├── compliance.py # Fine-tuned violation classifier
├── compliance_lora/ # Trained LoRA adapter weights (~15MB)
│ ├── adapter_config.json
│ └── adapter_model.safetensors
│
├── fine_tuning/ # Model training (one-time setup)
│ ├── generate_dataset.py # Creates 200 synthetic legal clauses
│ ├── train_lora.py # QLoRA training script (run on Colab GPU)
│ └── dataset/
│ └── compliance_data.json # Labeled training data
│
├── serving/
│ ├── init.py
│ └── api.py # FastAPI endpoints (/ingest, /query, /audit, /agent)
│
├── tests/ # Test suite
│ ├── ingest_test_data.py # Load documents into Qdrant
│ ├── test_rag.py # RAG pipeline tests
│ ├── test_audit.py # Compliance audit tests
│ └── test_agent.py # Agent workflow tests
│
├── docker-compose.yml # Qdrant vector database
├── .env # API keys and configuration
└── README.md


---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop (for Qdrant)
- Groq API key (free tier available)

### Setup
# 1. Clone and navigate
cd 07-enterprise-compliance-llm

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Add your Groq API key to .env

# 5. Start Qdrant
docker-compose up -d

# 6. Start API server
python serving/api.py

# In another terminal
python tests/ingest_test_data.py

# Test RAG (Question Answering)
python tests/test_rag.py

# Test Compliance Audit
python tests/test_audit.py

# Test AI Agent
python tests/test_agent.py
📡 API Endpoints
Endpoint	Method	Description	Example
/health	GET	Health check	curl http://localhost:8001/health
/ingest	POST	Upload document	curl -F "file=@contract.pdf" /ingest
/query	POST	Ask question	curl -d '{"question":"What is liability cap?"}' /query
/audit	POST	Check compliance	curl -F "file=@contract.pdf" /audit
/agent	POST	Multi-step workflow	curl -d '{"request":"Audit contracts and email report to me@email.com"}' /agent


🛠️ Tech Stack
Component	Technology	Purpose
LLM Inference	Groq (Llama-3.1-8B)	Fast, free-tier API
Fine-tuned Model	Llama-3.2-1B + LoRA	Compliance classification
Vector Database	Qdrant	Semantic search
Embeddings	all-MiniLM-L6-v2	384-dim sentence vectors
Backend	FastAPI + Uvicorn	REST API server
Training	Google Colab (T4 GPU)	Free GPU fine-tuning
Agent Framework	Custom ReAct implementation	Multi-step orchestration
Email	SMTP (Gmail/Outlook)	Report delivery


🎓 Concepts Demonstrated

# Milestone 1: RAG Pipeline
Document parsing (PDF, DOCX, TXT)

Semantic chunking with overlap strategy

Vector embeddings and similarity search

Prompt engineering for grounded answers

Context window management


# Milestone 2: Fine-Tuned Classifier
QLoRA fine-tuning (4-bit quantization)

LoRA adapters (only 0.07% parameters trained)

Synthetic training data generation

Multi-class compliance classification (GDPR, SOX, SAFE, NEEDS_REVIEW)

Severity scoring (HIGH, MEDIUM, LOW)


# Milestone 3: LLM Agents
ReAct pattern (Reasoning + Acting)

Tool definition and function calling

Multi-step workflow orchestration

State management across steps

Error handling and recovery

Real email integration via SMTP


# Milestone 4: Production Deployment, Security & LLMOps

Transform the development prototype into a production-ready system with monitoring, security, and observability.

---

## 🎯 What You Built

A production-grade LLM operations layer that tracks every request, detects security threats, and visualizes system health in real-time.

---

## 🧠 Concepts Learned 

### 1. Singleton Pattern
**What:** Ensures only ONE instance of a class exists across the entire application.

**Example:**
# Without Singleton (Milestone 1-3):
monitor1 = LLMMonitor()  # Terminal 1: has data
monitor2 = LLMMonitor()  # Terminal 2: empty - DIFFERENT object

# With Singleton (Milestone 4):
monitor1 = LLMMonitor()  # Returns the SAME object
monitor2 = LLMMonitor()  # Returns the SAME object
*Both see the same data!


## .env Example

# LLM Provider (Groq - free tier)
LLM_PROVIDER=groq
LLM_API_KEY=gsk_your_key_here
LLM_MODEL=llama-3.1-8b-instant

# Embeddings (local)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=compliance_docs

# Email (optional - for real email sending)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password


📁 Files Created This Milestone
File	Purpose	Key Concept
security/input_guard.py	Detect prompt injection	Security patterns
security/output_filter.py	Redact PII from responses	Regex matching
security/audit_logger.py	Record all requests/responses	Audit trail
llmops/monitor.py	Track performance metrics	Singleton pattern
llmops/eval_harness.py	Automated accuracy testing	Test automation
llmops/test_cases.json	10 test cases with criteria	Regression testing
llmops/prompts/v1.yaml	Version-controlled prompts	Configuration management
llmops/dashboard/app.py	Real-time monitoring UI	Streamlit
deployment/Dockerfile.api	Container recipe for API	Containerization
deployment/Dockerfile.vllm	Container recipe for vLLM	Containerization
deployment/docker-compose.prod.yml	Production stack	Orchestration
deployment/nginx.conf	Reverse proxy config	Rate limiting
configs/model_config.yaml	Model settings	Centralized config
configs/deployment_config.yaml	Server settings	Centralized config
.github/workflows/deploy.yml	CI/CD pipeline	Automation
tests/test_eval_harness.py	Test the test system	Meta-testing

🎓 Key Takeaways
Monitoring is not optional - You can't improve what you don't measure

Security must be layered - Input checks + output filters + audit logs

Percentiles matter more than averages - P95 tells you about user pain

Separate processes need shared storage - RAM isolation is why your dashboard was empty

YAML for config, .env for secrets - Never commit API keys

Automated evals catch regressions - 10 test cases run in seconds, manual testing takes hours

📜 License
MIT