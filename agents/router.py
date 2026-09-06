import os
from dotenv import load_dotenv

from langgraph.types import Send
from langchain.chat_models import init_chat_model
from core.state import RouterState, ClassificationResult

load_dotenv()
router_llm = init_chat_model(model="gemini-3.1-flash-lite", model_provider="google_genai")

def classify_query(state: RouterState) -> dict:
    """Classify query and determine which agents to invoke."""
    
    # 1. Give the router context about the uploaded file
    file_path = state.get("file_path")
    if file_path:
        filename = os.path.basename(file_path)
        doc_context = f"The user has uploaded a file named '{filename}'."
    else:
        doc_context = "The user has NOT uploaded any files."

    # 2. Update the system prompt with strict routing rules
    system_prompt = f"""Analyze this query and determine which knowledge bases to consult.
For each relevant source, generate a targeted sub-question optimized for that source.

{doc_context}

Available sources:
- code: Code examples, API references, software implementation details.
- search: Web search, recent information, and general knowledge.
- document: Use this source ONLY if the query explicitly mentions reading a file/document, OR if the query asks about specific data that logically aligns with the uploaded file's name.

Crucial Routing Rule:
If the user's query is about general technology (like "React hooks" or "Python syntax") and has no obvious connection to the uploaded file, DO NOT route to the document agent. Skip it entirely and only route to code or search.
"""

    structured_llm = router_llm.with_structured_output(ClassificationResult)
    
    result = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["query"]}
    ])
    
    print("\n=== CLASSIFICATION DECISIONS ===")
    for c in result.classifications:
        print(f"Source: {c.source}")
        print(f"Query : {c.query}")
        print("-" * 40)
    return {"classifications": result.classifications}

def route_to_agents(state: RouterState) -> list[Send]:
    """Fan out to agents based on classifications."""
    return [
        Send(c.source, {
            "query": c.query, 
            "file_path": state.get("file_path")
        })
        for c in state["classifications"]
    ]