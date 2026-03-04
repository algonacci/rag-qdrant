import os
import glob
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama

# --- CONFIGURATION ---
COLLECTION_NAME = "my_documents"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "nomic-embed-text"
INPUT_FOLDER = "input_data"
CHUNK_SIZE = 2000  # Karakter per chunk (sekitar 300-500 kata)
CHUNK_OVERLAP = 200 # Overlap biar konteksnya gak terputus

def get_embedding(text):
    """Get embedding from Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response['embedding']

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple recursive-ish chunking without LangChain."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def ingest_pdfs():
    # 1. Initialize Qdrant Client (disable check biar gak cerewet soal versi)
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    
    # 2. Ensure collection exists
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    # 3. Find PDF files
    pdf_files = glob.glob(os.path.join(INPUT_FOLDER, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {INPUT_FOLDER} folder.")
        return

    points = []
    point_id_counter = 1

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"Processing: {file_name}...")
        
        reader = PdfReader(pdf_path)
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text.strip():
                continue
                
            # Pecah text per halaman jadi chunk kecil
            page_chunks = chunk_text(text)
            
            for j, chunk in enumerate(page_chunks):
                print(f"  - Embedding page {i+1} (chunk {j+1})...")
                vector = get_embedding(chunk)
                
                points.append(PointStruct(
                    id=point_id_counter,
                    vector=vector,
                    payload={
                        "file_name": file_name,
                        "page": i + 1,
                        "chunk": j + 1,
                        "content": chunk
                    }
                ))
                point_id_counter += 1

    # 4. Upsert to Qdrant (dengan batching biar gak kena payload limit)
    if points:
        BATCH_SIZE = 100  # Kirim 100 data per request biar gak kegedean
        print(f"Upserting {len(points)} points to Qdrant in batches...")
        
        for k in range(0, len(points), BATCH_SIZE):
            batch = points[k : k + BATCH_SIZE]
            print(f"  - Sending batch {k // BATCH_SIZE + 1}...")
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
        
        print("Ingestion complete!")
    else:
        print("No content found to ingest.")

if __name__ == "__main__":
    ingest_pdfs()
