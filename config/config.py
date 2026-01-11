import os
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")
def llm_caller():
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    return llm