from langgraph.types import Send
from langchain.chat_models import init_chat_model
from core.state import RouterState, ClassificationResult

router_llm = init_chat_model(model="gemini-3.1-flash-lite", model_provider="google_genai")

def classify_query(state: RouterState) -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = router_llm.with_structured_output(ClassificationResult)
    
    result = structured_llm.invoke([
        {
            "role": "system",
            "content": "Analyze this query and determine which knowledge bases to consult.\nFor each relevant source, generate a targeted sub-question optimized for that source.\n\nAvailable sources:\n- code: Code examples, API references, implementation details\n- search: Web search and official documentation"
        },
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
        Send(c.source, {"query": c.query})
        for c in state["classifications"]
    ]
