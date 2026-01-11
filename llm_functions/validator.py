from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import PromptTemplate
from config.config import llm_caller

# ---------- Structured Output Model ----------
class RetrievalValidationOutput(BaseModel):
    needs_retrieval: str = Field(
        description="Whether new retrieval is needed. False if retrieved docs sufficiently answer the query."
    )
    new_query: Optional[str] = Field(
        default=None,
        description="A refined or more specific query to retrieve missing information (only if needs_retrieval is True)."
    )

# ---------- LLM-based Validation Function ----------
def validate_retrieval(state_partial):
    """
    Uses an LLM to determine whether the retrieved documents sufficiently answer the query.
    If not, it returns that new retrieval is needed along with a refined query.
    """

    docs_text = "\n\n".join(doc.get("text", "") for doc in state_partial['documents'] if isinstance(doc, dict))

    # Define LLM prompt
    validation_prompt = PromptTemplate.from_template(
        """
        You are a careful retrieval validation assistant.
        Your task is to determine whether the retrieved documents provide a **complete and correct answer** to the user query.

        INSTRUCTIONS:
        - Carefully read the retrieved documents and compare them to the query.
        - If the retrieved information is enough to fully and correctly answer the query, set "needs_retrieval" to false.
        - If the information is incomplete, missing, or off-topic, set "needs_retrieval" to true and propose a refined query that could retrieve the missing details.

        CONTEXT:
        User Query:
        {query}

        Retrieved Documents:
        {docs_text}

        Respond strictly in this JSON format:
        {{
            "needs_retrieval": 'true' or 'false',
            "new_query": "a refined retrieval query if true, otherwise null"
        }}
        """
    )

    llm = llm_caller()
    structured_llm_router = llm.with_structured_output(RetrievalValidationOutput)
    validation_chain = validation_prompt | structured_llm_router

    parsed = validation_chain.invoke({"query": state_partial['main_query'], "docs_text": docs_text})
    return parsed
