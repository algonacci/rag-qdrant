import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

EMBED_MODEL = "nomic-embed-text"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_documents"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

def ingest_document(file_path: str):
    loader = PyPDFLoader(file_path) if file_path.endswith('.pdf') else TextLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)

    # Tambahkan metadata source yang lebih informatif
    for i, doc in enumerate(splits):
        doc.metadata["source"] = file_path
        doc.metadata["chunk_index"] = i

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)

    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client,
    )

    print(f"✅ Ingested {len(splits)} chunks from {file_path}")

def search_qdrant(client, question: str, k: int = 4) -> list[Document]:
    """Search Qdrant directly and return LangChain Documents with proper metadata."""
    import ollama

    # Get embedding for the question
    response = ollama.embeddings(model=EMBED_MODEL, prompt=question)
    query_vector = response['embedding']

    # Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
    ).points

    # Convert to LangChain Documents
    docs = []
    for hit in results:
        payload = hit.payload
        doc = Document(
            page_content=payload.get("content", ""),
            metadata={
                "file_name": payload.get("file_name"),
                "page": payload.get("page"),
                "chunk": payload.get("chunk"),
                "score": hit.score,
            }
        )
        docs.append(doc)

    return docs

def query_rag(question: str):
    client = QdrantClient(url=QDRANT_URL)

    # Use custom search function
    docs = search_qdrant(client, question, k=4)
    
    # Format context from retrieved docs
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL_NAME"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context below:

    Context: {context}

    Question: {question}

    Answer:
    """)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, docs

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run rag.py ingest <file_path>")
        print("  uv run rag.py query <question>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "ingest":
        ingest_document(sys.argv[2])
    elif command == "query":
        answer, docs = query_rag(sys.argv[2])
        print(f"\n🤖 Answer: {answer}")
        print(f"\n📄 Retrieved Documents ({len(docs)}):")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Document {i} ---")
            meta = doc.metadata
            # Support format dari ingest.py (file_name, page, chunk)
            source = meta.get('file_name') or meta.get('source', 'N/A')
            print(f"Source: {source}")
            if 'score' in meta:
                print(f"Score: {meta['score']:.4f}")
            if 'page' in meta:
                print(f"Page: {meta['page']}")
            if 'chunk' in meta:
                print(f"Chunk: {meta['chunk']}")
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            print(f"Content:\n{content_preview}")
    else:
        print(f"Unknown command: {command}")
