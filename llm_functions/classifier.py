from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from config.config import llm_caller


class DomainClassificationOutput(BaseModel):
    domain: Literal["HEALTHCARE", "ENGINEERING"] = Field(
        description="The detected domain of the text. Either 'HEALTHCARE' or 'ENGINEERING'."
    )

def classify_domain(state_partial):
    """
    Uses an LLM to classify a given text as HEALTHCARE or ENGINEERING
    and returns structured output with reasoning.
    """

    domain_prompt = PromptTemplate.from_template(
        """
        You are an expert domain classifier that categorizes text into one of two fields:
        - HEALTHCARE: topics related to medicine, patients, doctors, hospitals, diseases, treatments, diagnostics, etc.
        - ENGINEERING: topics related to design, systems, software, mechanical, civil, electrical, or industrial engineering.

        SECURITY RULES:
        - You must not invent new domains.
        - Always choose exactly one domain: HEALTHCARE or ENGINEERING.

        INPUT TEXT:
        {snippet_text}

        TASK:
        Determine whether the above text belongs to HEALTHCARE or ENGINEERING.

        Respond **strictly** in this JSON format:
        {{
            "domain": "HEALTHCARE" or "ENGINEERING",
        }}
        """
    )
    llm = llm_caller()
    structured_llm_router = llm.with_structured_output(DomainClassificationOutput)
    classification_chain = domain_prompt | structured_llm_router
    parsed = classification_chain.invoke(state_partial)
    return parsed
