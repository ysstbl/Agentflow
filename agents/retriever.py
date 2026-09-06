from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from core.state import AgentInput
from tools.web_search import tavily_search, retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


load_dotenv()

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



def query_code(state: AgentInput) -> dict:
    result = code_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "code", "result": result["messages"][-1].content}]}

def query_search(state: AgentInput) -> dict:
    result = web_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "search", "result": result["messages"][-1].content}]}

def query_document(state: AgentInput) -> dict:
    file_path = state.get("file_path")
    
    if not file_path:
        return {"results": [{"source": "document", "result": "Error: No file path provided in state."}]}

    # 1. Parse the raw binary PDF into human-readable text
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        document_text = "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return {"results": [{"source": "document", "result": f"Failed to read PDF: {str(e)}"}]}

    # 2. Inject the extracted text directly into the prompt
    prompt = f"""
    You are an expert document analyst. Use the following document to answer the user's query.
    If the answer is not in the document, say so.
    
    DOCUMENT CONTENT:
    {document_text}
    
    USER QUERY:
    {state['query']}
    """

    # 3. Call your router_llm directly (bypassing the old document_agent)
    result = router_llm.invoke(prompt)
    
    return {"results": [{"source": "document", "result": result.content}]}