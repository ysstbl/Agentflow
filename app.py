!pip install -qU \
langchain \
langchain-core \
langchain-community \
langchain-google-genai \
langgraph \
langchain-huggingface \
tavily-python \
markdownify

import os
import operator
from typing import Annotated, Literal, TypedDict

import httpx
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langchain.tools import InjectedToolArg, tool
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send



# Set environment variables
os.environ["GOOGLE_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

# Router LLM
router_llm = init_chat_model(model="gemini-3.1-flash-lite", model_provider="google_genai")

# Class definitions
class Classification(BaseModel):
    """A single routing decision: which agent to call with what query."""
    source: Literal["code", "search", "document"]
    query: str

class AgentInput(TypedDict):
    """Simple input state for each subagent."""
    query: str

class AgentOutput(TypedDict):
    """Output from each subagent."""
    source: str
    result: str

class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]  # Reducer collects parallel results
    final_answer: str

# Tavily client
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Helper function
def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch webpage and convert HTML to markdown."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching {url}: {e!s}"

# Tool definitions
@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns full webpage content as markdown.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with full webpage content
    """
    search_results = tavily_client.search(
        query=query,
        search_depth="advanced"
    )
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        content = result["content"]
        result_texts.append(f"## {title}\n**URL:** {url}\n\n{content}\n---")

    return f"Found {len(result_texts)} result(s) for '{query}':\n\n" + "\n".join(
        result_texts
    )



def retrieve_docs(query: str):
    # If vector_store is not set up, this function will return a placeholder.
    # docs = retriever.invoke(query)
    # return "\n\n".join([doc.page_content for doc in docs])
    return "Document retrieval is not yet implemented. Please set up 'vector_store' for document retrieval."


retriever_tool = Tool(
    name="WebTechnologiesSearch",
    func=retrieve_docs,
    description="Search the Web Technologies PDF for relevant information."
)



# Agent definitions for the workflow
web_agent = create_agent(
    router_llm,
    tools=[tavily_search],
    system_prompt=(
        """ You are a search assistant conducting search on the user's input topic. For context, today's date is {date}.

Your job is to use tools to gather information about the user's input topic.
You can use the tavily_search tool to find resources that can help answer the research question.
You can call it in series or in parallel, your research is conducted in a tool-calling loop.

You have access to the tavily_search tool for conducting web searches.

Think like a human researcher with limited time."""
    ),
)
code_agent = create_agent(
    router_llm,
    tools=[tavily_search],
    system_prompt="""
    You are an AI coding assistant.

    Your job:
    - Answer programming questions
    - Search the web when needed
    - Generate clean example code
    - Explain the code clearly
    - Prefer practical examples
    - Return runnable code snippets
    """
)

document_agent = create_agent(
    router_llm,
    tools=[retriever_tool],
    system_prompt="""
    Retrieve information from the source based on the given query
    """
)

# Define structured output schema for the classifier
class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )

# Graph nodes/functions
def classify_query(state: RouterState) -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = router_llm.with_structured_output(ClassificationResult)

    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """Analyze this query and determine which knowledge bases to consult.
For each relevant source, generate a targeted sub-question optimized for that source.

Available sources:
- code: Code examples, API references, implementation details
- search: Web search and official documentation """
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

def query_code(state: AgentInput) -> dict:
    """Query the coding agent."""
    result = code_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "web", "result": result["messages"][-1].content}]}

def query_search(state: AgentInput) -> dict:
    """Query the search agent."""
    result = web_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "web", "result": result["messages"][-1].content}]}

def query_document(state: AgentInput) -> dict:
    """Query the document agent."""
    result = document_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "document", "result": result["messages"][-1].content}]}

def synthesize_results(state: RouterState) -> dict:
    """Combine results from all agents into a coherent answer."""
    if not state["results"]:
        return {"final_answer": "No results found from any knowledge source."}

    formatted = [
        f"**From {r['source'].title()}:**\n{r['result']}"
        for r in state["results"]
    ]

    synthesis_response = router_llm.invoke([
        {
            "role": "system",
            "content": f"""Synthesize these search results to answer the original question: "{state['query']}"\n\n- Combine information from multiple sources without redundancy\n- Highlight the most relevant and actionable information\n- Note any discrepancies between sources\n- Keep the response concise and well-organized"""
        },
        {"role": "user", "content": "\n\n".join(formatted)}
    ])
    content = synthesis_response.content
    if isinstance(content, list):
        content = content[0]["text"] # Assuming content might be a list of dicts

    return {"final_answer": content}

# Build the LangGraph workflow
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
    .compile()
)

result = workflow.invoke({"query": "what are react hooks"})
print(result["final_answer"])
