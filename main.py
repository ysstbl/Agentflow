import os
from graph import build_graph

# Setup your API keys here
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"

def main():
    print("Initializing AgentFlow...")
    app = build_graph()
    
    query = "what are react hooks"
    print(f"\nExecuting Query: '{query}'\n")
    
    result = app.invoke({"query": query})
    
    print("\n=== FINAL ANSWER ===\n")
    print(result["final_answer"])

if __name__ == "__main__":
    main()
