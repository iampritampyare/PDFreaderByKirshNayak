from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Test FAISS Index Creation
db = FAISS.from_texts(["MCQ 1: What is AI?", "MCQ 2: What is Machine Learning?"], embedding=embeddings)

# Save to local storage
db.save_local("faiss_index")

print("FAISS index saved successfully!")
