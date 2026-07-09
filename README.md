# AgentFlow 🧠

AgentFlow is a modular, multi-agent AI workflow built with Python and LangGraph. It intelligently classifies user queries, routes them to specialized AI sub-agents (Code, Web Search, and Document Retrieval), executes searches in parallel, and synthesizes the findings into a single, cohesive response.

## 🚀 Features

* **Intelligent Routing:** Uses a Gemini 3.1 Flash-Lite router to dynamically analyze and break down complex queries.
* **Parallel Agent Execution:** Leverages LangGraph for fanning out tasks to sub-agents simultaneously, significantly reducing response latency.
* **Live Web Integration:** Utilizes Tavily API to fetch and parse real-time web data beyond standard LLM training cutoffs.
* **Structured State Management:** Enforces strict data contracts between agents using Pydantic models.
* **Modular Architecture:** Clean separation of concerns (Core State, Tools, Agents, and Graph logic) designed for scalability.

## 🏗️ Architecture 

The workflow follows a directed graph topology:

1.  **Classify Node:** Evaluates the query and determines which data sources (Code, Search, Document) are required.
2.  **Route:** Conditionally fans out the query to the chosen sub-agents.
3.  **Retrieve:** Agents fetch data independently (using tools like Tavily).
4.  **Synthesize Node:** Waits for all parallel processes to complete, aggregates the `AgentOutput` states, and generates a final, non-redundant answer.

## 📁 Project Structure

```text
agentflow/
├── agents/                 
│   ├── router.py           # Query classification and routing logic
│   ├── retriever.py        # Specialized sub-agents (Code, Search, Document)
│   └── synthesizer.py      # Final response aggregation
├── tools/                  
│   └── web_search.py       # Tavily API integration and Markdown parsing
├── core/                   
│   └── state.py            # Pydantic models and TypedDicts for Graph state
├── graph.py                # LangGraph orchestration and node compilation
├── main.py                 # Application entry point
└── requirements.txt        # Python dependencies
