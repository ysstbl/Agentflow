from langgraph.graph import StateGraph, START, END
from core.state import RouterState
from agents.router import classify_query, route_to_agents
from agents.retriever import query_code, query_search, query_document
from agents.synthesizer import synthesize_results

def build_graph():
    """Compiles and returns the LangGraph workflow."""
    workflow = (
        StateGraph(RouterState)
        .add_node("classify", classify_query)
        .add_node("code", query_code)
        .add_node("search", query_search)
        .add_node("document", query_document)
        .add_node("synthesize", synthesize_results)
        .add_edge(START, "classify")
        .add_conditional_edges("classify", route_to_agents, ["code", "search", "document"])
        .add_edge("code", "synthesize")
        .add_edge("search", "synthesize")
        .add_edge("document", "synthesize")
        .add_edge("synthesize", END)
    )
    return workflow.compile()
