"""
Simple PDF knowledge base for Alzheimer's research papers.
Extracts text from PDFs, chunks it, and finds relevant chunks for a query.
No heavy dependencies - just PyPDF2 and keyword matching.
"""
import os
import re
import json
import pickle
from pathlib import Path

PDF_DIR = "/root/LocalGPT-Chatbot/uploads"
CACHE_FILE = "/root/LocalGPT-Chatbot/backend/pdf_chunks.pkl"
CHUNK_SIZE = 800   # characters per chunk
CHUNK_OVERLAP = 150

def extract_pdf_text(pdf_path):
    """Extract all text from a PDF file."""
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    text += page.extract_text() + "\n"
                except:
                    continue
        return text
    except Exception as e:
        print(f"[ERROR] Failed to extract {pdf_path}: {e}")
        return ""

def chunk_text(text, source_name):
    """Split text into overlapping chunks."""
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if len(chunk) > 100:
            chunks.append({
                'text': chunk,
                'source': source_name
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build_index():
    """Extract text from all PDFs and build chunk index."""
    print("Building PDF knowledge base...")
    all_chunks = []
    for pdf_file in sorted(os.listdir(PDF_DIR)):
        if not pdf_file.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"  Processing: {pdf_file}")
        text = extract_pdf_text(pdf_path)
        clean_name = pdf_file.replace('TODAY', '').replace('.pdf', '')
        chunks = chunk_text(text, clean_name)
        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} chunks")

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(all_chunks, f)
    print(f"\nTotal: {len(all_chunks)} chunks from {len(os.listdir(PDF_DIR))} PDFs")
    print(f"Saved to: {CACHE_FILE}")
    return all_chunks

def load_index():
    """Load cached chunks."""
    if not os.path.exists(CACHE_FILE):
        return build_index()
    with open(CACHE_FILE, 'rb') as f:
        return pickle.load(f)

def search(query, top_k=5):
    """Find most relevant chunks using keyword scoring."""
    chunks = load_index()
    query_words = set(re.findall(r'\w+', query.lower()))
    query_words = {w for w in query_words if len(w) > 3}

    if not query_words:
        return []

    scored = []
    for chunk in chunks:
        text_lower = chunk['text'].lower()
        score = sum(text_lower.count(word) for word in query_words)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

def get_context_for_query(query, max_chars=3000):
    """Get formatted context from relevant PDFs for a query."""
    results = search(query, top_k=5)
    if not results:
        return ""

    context_parts = []
    total_chars = 0
    for r in results:
        chunk_text = f"[Source: {r['source']}]\n{r['text']}\n"
        if total_chars + len(chunk_text) > max_chars:
            break
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n---\n".join(context_parts)

if __name__ == "__main__":
    build_index()
    print("\n--- Testing search ---")
    results = search("APOE4 gene Alzheimer's risk", top_k=3)
    for r in results:
        print(f"\nSource: {r['source']}")
        print(f"Text: {r['text'][:200]}...")
