import os
import httpx
from markdownify import markdownify
from typing import Annotated, Literal
from tavily import TavilyClient
from langchain.tools import InjectedToolArg, tool
from langchain_core.tools import Tool

# Initialize client (ensure env vars are set in main.py or your environment)
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))

def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch webpage and convert HTML to markdown."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching {url}: {e!s}"

@tool
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Search the web for information on a given query."""
    search_results = tavily_client.search(query=query, search_depth="advanced")
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        content = result["content"]
        result_texts.append(f"## {title}\n**URL:** {url}\n\n{content}\n---")

    return f"Found {len(result_texts)} result(s) for '{query}':\n\n" + "\n".join(result_texts)

def retrieve_docs(query: str):
    return "Document retrieval is not yet implemented. Please set up 'vector_store' for document retrieval."

retriever_tool = Tool(
    name="WebTechnologiesSearch",
    func=retrieve_docs,
    description="Search the Web Technologies PDF for relevant information."
)
