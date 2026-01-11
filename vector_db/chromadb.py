import chromadb
import os
from utils.text_splitter import split_text
from utils.text_embeddings import get_embeddings

chromadb_path = os.getenv("CHROMADB_PATH")
chromadb_collection = os.getenv("CHROMADB_COLLECTION")

def get_chroma_collection(chromadb_collection):
    client = chromadb.PersistentClient(path=chromadb_path)
    try:
        collection = client.get_collection(name=chromadb_collection)
    except:
        collection = client.create_collection(name=chromadb_collection)
    return collection

def add_to_chroma(text, metadata):
    collection = get_chroma_collection(chromadb_collection)
    text_chunks = split_text(text)
    embeddings = get_embeddings(text_chunks)
    ids = [f"doc_{i}" for i in range(len(text_chunks))]
    metas = [metadata for _ in text_chunks]
    collection.add(documents=text_chunks, embeddings=embeddings.tolist(), ids=ids, metadatas=metas)
    print(f"✅ Added {len(text_chunks)} chunks to ChromaDB")

def query_chroma(query, top_k=3):
    collection = get_chroma_collection(chromadb_collection)
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]
    output = []
    for doc, meta, dist in zip(docs, metas, dists):
        output.append({
            "text": doc,
            "metadata": meta,
            "score": float(dist)
        })

    return output