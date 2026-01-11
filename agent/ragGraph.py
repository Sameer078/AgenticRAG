from typing import TypedDict, Literal, NotRequired, List
from langgraph.graph import END, StateGraph, START
from langchain_core.runnables import RunnableLambda
from llm_functions.classifier import classify_domain
from llm_functions.validator import validate_retrieval
from vector_db.chromadb import query_chroma
from vector_db.qdrant import query_qdrant
from llm_functions.generator import generate_rag_response

# ---------- Define State Types ----------
class ragGraphState(TypedDict):
    domain: NotRequired[Literal["HEALTHCARE", "ENGINEERING"]]
    answer: NotRequired[str]
    sources: List[str]
    main_query: NotRequired[str]
    current_query: NotRequired[str]
    documents: List[dict]
    needs_retrieval: NotRequired[str]

# ---------- Domain Detection (LLM with structured output) ----------
def detect_domain(state: ragGraphState) -> ragGraphState:
    state_partial = {
            "snippet_text": state["current_query"]
        }
    parsed = classify_domain(state_partial)
    return {**state, "domain": parsed.domain}

# ---------- Retrieve qdrant data ----------
def query_qdrantdb(state: ragGraphState) -> ragGraphState:
    parsed = query_qdrant(state["current_query"])
    print(parsed)
    return {**state, "documents": state.get("documents", []) + parsed}

# ---------- Retrieve chromadb data ----------
def query_chromadb(state: ragGraphState) -> ragGraphState:
    parsed = query_chroma(state["current_query"])
    print(parsed)
    return {**state, "documents": state.get("documents", []) + parsed}

# ---------- Domain Detection (LLM with structured output) ----------
def validate_docs(state: ragGraphState) -> ragGraphState:
    if state['documents']:
        state_partial = {
                "documents": state["documents"],
                "main_query":"main_query"
            }
        parsed = validate_retrieval(state_partial)
        print(parsed)
        return {**state, "needs_retrieval": parsed.needs_retrieval, "current_query": parsed.new_query}
    else:
        return {**state, "needs_retrieval": 'false'}

# ---------- Generate Response ----------
def generate_response(state: ragGraphState) -> ragGraphState:
    state_partial = {
            "documents": state["documents"],
            "main_query":"main_query"
        }
    parsed = generate_rag_response(state_partial)
    print(parsed)
    sources = [doc.get("metadata", {}).get("source", "unknown") for doc in state['documents']]
    return {**state, "answer": parsed.answer, "sources":list(set(sources))}


def graph_rag():
    agenticrag_graph = StateGraph(ragGraphState)
    
    agenticrag_graph.add_node("detect_domain", RunnableLambda(detect_domain))
    agenticrag_graph.add_node("query_qdrantdb", RunnableLambda(query_qdrantdb))
    agenticrag_graph.add_node("query_chromadb", RunnableLambda(query_chromadb))

    def vectordb_router(state: ragGraphState) -> dict:
        if state.get("domain") == "HEALTHCARE":
            return {"next": "qdrant"}
        else:
            return {"next": "chromadb"}

    agenticrag_graph.add_node("vectordb_router", RunnableLambda(vectordb_router))
    agenticrag_graph.add_node("validate_docs", RunnableLambda(validate_docs))
    agenticrag_graph.add_node("generate_response", RunnableLambda(generate_response))
    def validate_router(state: ragGraphState) -> dict:
        if state.get("needs_retrieval") == "true":
            return {"next": "true"}
        else:
            return {"next": "false"}
    
    agenticrag_graph.add_node("validate_router", RunnableLambda(validate_router))
    
    agenticrag_graph.add_edge(START, "detect_domain")
    agenticrag_graph.add_edge("detect_domain", "vectordb_router")

    agenticrag_graph.add_conditional_edges(
        "vectordb_router",
        lambda state: state.get("next"),
        {"qdrant": "query_qdrantdb", "chromadb": "query_chromadb"},
    )

    agenticrag_graph.add_edge("query_qdrantdb", "validate_docs")
    agenticrag_graph.add_edge("query_chromadb", "validate_docs")
    
    agenticrag_graph.add_edge("validate_docs", "validate_router")

    agenticrag_graph.add_conditional_edges(
        "validate_router",
        lambda state: state.get("next"),
        {"true": "detect_domain", "false": "generate_response"},
    )

    agenticrag_graph.add_edge("generate_response", END)
    agentRagRun = agenticrag_graph.compile()
    return agentRagRun
