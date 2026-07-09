from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from core.state import AgentInput
from tools.web_search import tavily_search, retriever_tool

router_llm = init_chat_model(model="gemini-3.1-flash-lite", model_provider="google_genai")

web_agent = create_agent(
    router_llm,
    tools=[tavily_search],
    system_prompt="You are a search assistant conducting search on the user's input topic. Your job is to use tools to gather information about the user's input topic. You can use the tavily_search tool to find resources. Think like a human researcher with limited time."
)

code_agent = create_agent(
    router_llm,
    tools=[tavily_search],
    system_prompt="You are an AI coding assistant.\nYour job:\n- Answer programming questions\n- Search the web when needed\n- Generate clean example code\n- Explain the code clearly\n- Prefer practical examples\n- Return runnable code snippets"
)

document_agent = create_agent(
    router_llm,
    tools=[retriever_tool],
    system_prompt="Retrieve information from the source based on the given query"
)

def query_code(state: AgentInput) -> dict:
    result = code_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "code", "result": result["messages"][-1].content}]}

def query_search(state: AgentInput) -> dict:
    result = web_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "search", "result": result["messages"][-1].content}]}

def query_document(state: AgentInput) -> dict:
    result = document_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "document", "result": result["messages"][-1].content}]}
