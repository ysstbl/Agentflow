import os
from dotenv import load_dotenv
from graph import build_graph

# 1. Load the variables from .env
load_dotenv()

def main():
    print("Initializing AgentFlow...")
    app = build_graph()
    
    # 2. Construct the absolute path to your PDF
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    target_pdf_path = os.path.join(BASE_DIR, "sample_pdf", "sample1.pdf")
    
    if not os.path.exists(target_pdf_path):
        print(f"Warning: Could not find PDF at {target_pdf_path}. Document queries will fail.")

    # 3. Use a compound query to trigger ALL agents (Document, Search, Code)
    query = (
        "What are React hooks? Explain with a short code example."
    )
    
    print(f"\nExecuting Query: '{query}'\n")
    
    # 4. Run the graph
    result = app.invoke({
        "query": query,
        "file_path": target_pdf_path
    })
    
    print("\n=== FINAL ANSWER ===\n")
    print(result.get("final_answer", "No final answer returned."))

if __name__ == "__main__":
    main()