import os
from dotenv import load_dotenv

# 1. Load environment variables from .env
load_dotenv()

# 2. Correctly import from the agents folder
from agents.retriever import query_document

# 3. Point to your PDF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# IMPORTANT: ensure "your_folder_name" matches exactly where sample.pdf is
pdf_path = os.path.join(BASE_DIR, "sample_pdf", "sample1.pdf")

# 4. Create a test payload asking something specific to that PDF
test_state = {
    "file_path": pdf_path,
    "query": "Give me a 2-sentence summary of what this document is about."
}

# 5. Run the function
print(f"Testing query_document with file: {pdf_path}...\n")
output = query_document(test_state)

print("=== Result ===")
print(output["results"][0]["result"])