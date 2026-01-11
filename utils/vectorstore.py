from llm_functions.classifier import classify_domain
from vector_db.chromadb import add_to_chroma
from vector_db.qdrant import add_to_qdrant

def add_text(text, metadata):
    """Auto-classify domain and add text to chosen vector database."""
    output = classify_domain({"snippet_text":text})
    print(output.domain)
    if output.domain == "HEALTHCARE":
        add_to_qdrant(text, metadata)
    elif output.domain == "ENGINEERING":
        add_to_chroma(text, metadata)
    else:
        print("invalid content")
    return "SUCCESS"
