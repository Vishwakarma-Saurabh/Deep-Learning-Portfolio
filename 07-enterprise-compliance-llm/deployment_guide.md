# Deployment Guide

Deploy your Compliance AI Assistant to the cloud for public access.

---

## Option 1: Streamlit Cloud (Free - Recommended)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
Step 2: Deploy on Streamlit Cloud
Go to https://streamlit.io/cloud

Sign in with GitHub

Click "New app"

Select your repository

Set main file path: app.py

Click "Deploy"

Step 3: Set Secrets
In Streamlit Cloud dashboard → Settings → Secrets:

toml
LLM_API_KEY = "gsk_your_groq_key"
QDRANT_HOST = "your-qdrant-cloud-host"
QDRANT_PORT = "6333"
QDRANT_COLLECTION = "compliance_docs"
Step 4: Update API URL
Change in utils/api_client.py:

python
API_URL = "https://your-api-url.com"  # Deploy API separately or use localhost for demo
Option 2: Docker Deployment
Build and Run
bash
# Build API image
docker build -f deployment/Dockerfile.api -t compliance-api .

# Run with docker-compose
docker-compose -f deployment/docker-compose.prod.yml up -d

# Build Streamlit image
docker build -f deployment/Dockerfile.streamlit -t compliance-ui .

# Run Streamlit
docker run -p 8501:8501 compliance-ui
Option 3: Hugging Face Spaces
Go to https://huggingface.co/spaces

Create new Space

Choose Streamlit SDK

Push your code

Set environment variables in Settings

Environment Variables
Variable	Description	Required
LLM_API_KEY	Groq API key	Yes
QDRANT_HOST	Qdrant server host	Yes
QDRANT_PORT	Qdrant server port	Yes
SMTP_EMAIL	Email for sending reports	No
SMTP_PASSWORD	Email app password	No
Post-Deployment Checklist
API health check passes

Document upload works

Chat responses return correctly

Compliance audit functions

Agent workflows complete

Dashboard shows metrics

Rate limiting active

SSL/HTTPS enabled (if custom domain)