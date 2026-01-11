from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from uuid import uuid4
from utils.text_splitter import split_text
from utils.text_embeddings import get_embeddings
import os

qdrant_path = os.getenv("QDRANT_PATH")
qdrant_collection = os.getenv("QDRANT_COLLECTION")
dim = os.getenv("EMBEDDINGS_DIM")

def load_or_create_qdrant_collection(collection_name, dim=dim, qdrant_path=qdrant_path):
    client = QdrantClient(path=qdrant_path)
    existing_collections = [c.name for c in client.get_collections().collections]

    if collection_name not in existing_collections:
        print(f"Creating new Qdrant collection '{collection_name}'...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        print(f"Loaded existing Qdrant collection '{collection_name}'.")
    return client

def add_to_qdrant(text, metadata=None):
    text_chunks = split_text(text)
    embeddings = get_embeddings(text_chunks)
    points = []

    for i, emb in enumerate(embeddings):
        payload = {"text": text_chunks[i]}
        if metadata:
            payload.update(metadata)
        points.append(PointStruct(id=str(uuid4()), vector=emb, payload=payload))

    client = load_or_create_qdrant_collection(qdrant_collection, dim, qdrant_path)
    client.upsert(collection_name=qdrant_collection, points=points)
    print(f"✅ Added {len(text_chunks)} chunks to Qdrant collection '{qdrant_collection}'")

def query_qdrant(query, top_k=3):
    query_vec = get_embeddings([query])[0]
    client = load_or_create_qdrant_collection(qdrant_collection, dim, qdrant_path)
    hits = client.search(collection_name=qdrant_collection, query_vector=query_vec, limit=top_k)
    results = []

    for hit in hits:
        payload = hit.payload
        text = payload.get("text", "")
        metadata = {k: v for k, v in payload.items() if k != "text"}
        results.append({'text': text, 'metadata': metadata,'score': float(hit.score)})

    return results
