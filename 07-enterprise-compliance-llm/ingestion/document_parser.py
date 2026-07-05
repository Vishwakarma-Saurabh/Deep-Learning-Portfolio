"""
Document Parser - Extracts text from PDF and DOCX files.
Concept: Data preprocessing for RAG :- quality input = quality retrieval.
"""

import re
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document

def parse_document(file_path: str) -> dict:
 
    path = Path(file_path)
    
    if not path.exists():
        return {"text": "", "filename": path.name, "error": "File not found"}
    
    if path.suffix.lower() == '.pdf':
        text = _parse_pdf(path)
    elif path.suffix.lower() == '.docx':
        text = _parse_docx(path)
    else:
        return {"text": "", "filename": path.name, "error": "Unsupported format"}
    
    # Clean the text
    text = _clean_text(text)
    
    return {
        "text": text,
        "filename": path.name,
        "char_count": len(text),
        "status": "success"
    }


def _parse_pdf(path: Path) -> str:
    """Extract text from PDF using PyPDF2."""
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n\n".join(text_parts)


def _parse_docx(path: Path) -> str:
    """Extract text from DOCX using python-docx."""
    doc = Document(path)
    return "\n\n".join([para.text for para in doc.paragraphs])


def _clean_text(text: str) -> str:
    """Clean extracted text for better embeddings."""
    # Remove excessive newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove excessive spaces
    text = re.sub(r' +', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


# Test independently
if __name__ == "__main__":
    # Create a test file or use an existing one
    result = parse_document("sample-pdf.pdf")
    print(f"Parsed {result['filename']}: {result['char_count']} characters")
    print(f"First 200 chars: {result['text'][:200]}...")