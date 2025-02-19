import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize model once to avoid repeated API calls
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=0.3)

# Define Prompt
prompt_template = PromptTemplate(
    template="""
    Answer the question as detailed as possible from the provided context.
    If the answer is not present, just say 'Answer is not present in the context.'
    Do not provide incorrect information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
    input_variables=["context", "question"]
)

def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("FAISS index saved successfully!")

def get_retrieval_chain(retriever):
    """Create RetrievalQA chain with a custom prompt."""
    return RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}  # Attach prompt
    )

def user_input(user_question):
    """Handles user input and retrieves answers from the stored FAISS index."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    if not os.path.exists("faiss_index"):
        st.error("No FAISS index found! Please upload and process a PDF first.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    chain = get_retrieval_chain(retriever)

    # Debugging input
    print("DEBUG: Input to chain.invoke() ->", user_question, "Type:", type(user_question))

    try:
        response = chain.invoke({"query": user_question})  # Pass the correct input format
        print("DEBUG: Response:", response)
        st.write("Reply: ", response["result"])
    except Exception as e:
        print("ERROR:", e)
        st.error(f"Error: {e}")

def main():
    """Main function to run Streamlit UI."""
    st.set_page_config(page_title="Find all questions from the PDF")
    st.header("Find all MCQs from PDF using Gemini")
    
    user_question = st.text_input("Type your question about the PDF:")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit and Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done! PDF processed and indexed.")

if __name__ == "__main__":
    main()
