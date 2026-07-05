"""
Text Chunker - Splits documents into semantic chunks.
Concept: Chunking strategy directly impacts retrieval quality.
Too small = lost context, Too large = diluted relevance.
"""

from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:

    if not text:
        return []
    
    # First, split by paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    chunk_index = 0
    start_char = 0
    
    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            # Save current chunk
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": start_char + len(current_chunk)
            })
            
            # Start new chunk with overlap
            # Take last 'overlap' characters from previous chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            current_chunk = overlap_text + "\n\n" + para
            start_char = chunks[-1]["end_char"] - len(overlap_text)
            chunk_index += 1
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": start_char + len(current_chunk)
        })
    
    # Add metadata to each chunk
    for chunk in chunks:
        chunk["char_count"] = len(chunk["text"])
        # Rough token estimate (4 chars ≈ 1 token)
        chunk["approx_tokens"] = len(chunk["text"]) // 4
    
    return chunks
