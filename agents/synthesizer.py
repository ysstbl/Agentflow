from langchain.chat_models import init_chat_model
from core.state import RouterState

router_llm = init_chat_model(model="gemini-3.1-flash-lite", model_provider="google_genai")

def synthesize_results(state: RouterState) -> dict:
    """Combine results from all agents into a coherent answer."""
    if not state.get("results"):
        return {"final_answer": "No results found from any knowledge source."}

    formatted = [
        f"**From {r['source'].title()}:**\n{r['result']}"
        for r in state["results"]
    ]

    synthesis_response = router_llm.invoke([
        {
            "role": "system",
            "content": f"Synthesize these search results to answer the original question: \"{state['query']}\"\n\n- Combine information from multiple sources without redundancy\n- Highlight the most relevant and actionable information\n- Note any discrepancies between sources\n- Keep the response concise and well-organized"
        },
        {"role": "user", "content": "\n\n".join(formatted)}
    ])
    
    content = synthesis_response.content
    if isinstance(content, list):
        content = content[0]["text"] 

    return {"final_answer": content}
