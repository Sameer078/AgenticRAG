from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from config.config import llm_caller

class RAGResponseOutput(BaseModel):
    answer: str = Field(
        description="A coherent, informative response to the user's query using the provided documents."
    )



def generate_rag_response(state_partial):
    """
    Generate a response to the user's query based on retrieved documents (RAG).
    Returns structured output including answer and sources.
    """

    # Combine all texts and collect sources
    docs_text = "\n\n".join(doc.get("text", "") for doc in state_partial['documents'] if isinstance(doc, dict))

    rag_prompt = PromptTemplate.from_template(
        """
        You are a knowledgeable assistant tasked with answering the user's query
        using only the information provided in the documents below.

        USER QUERY:
        {main_query}

        DOCUMENTS:
        {docs_text}

        INSTRUCTIONS:
        - Use only the information from the documents; do not make up facts.
        - Keep the response clear, concise, and informative.

        Respond strictly in this JSON format:
        {{
            "answer": "The generated answer to the user's query.",
        }}
        """
    )

    llm = llm_caller()
    structured_llm_router = llm.with_structured_output(RAGResponseOutput)
    rag_chain = rag_prompt | structured_llm_router

    parsed = rag_chain.invoke({"main_query": state_partial['main_query'], "docs_text": docs_text})
    return parsed