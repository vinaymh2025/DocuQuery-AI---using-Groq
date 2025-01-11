import os
import tempfile

os.environ['USER_AGENT'] = 'MyCustomAgent/1.0'

import streamlit as st
import httpx
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import time
from dotenv import load_dotenv

load_dotenv()

# Load the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Please check your environment variables.")
    st.stop()

# Initialize session state variables
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if "loader" not in st.session_state:
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")

if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

if "vectors" not in st.session_state:
    embedding_dim = len(st.session_state.embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(embedding_dim)
    st.session_state.vectors = FAISS(
        embedding_function=st.session_state.embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

# Streamlit UI
st.title("DocuQuery AI - powered by Groq")

# File uploader for user documents
uploaded_file = st.file_uploader("Upload a file (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Determine and initialize the document loader based on file extension
    loader = PyPDFLoader(tmp_file_path) if uploaded_file.type == "application/pdf" else TextLoader(tmp_file_path)

    # Process and split uploaded documents
    new_docs = loader.load()
    split_docs = st.session_state.text_splitter.split_documents(new_docs)

    # Add documents to the vector store
    st.session_state.vectors.add_texts([doc.page_content for doc in split_docs])
    st.success("File uploaded and processed successfully!")

# Initialize the LLM
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
except Exception as e:
    st.error(f"Error initializing ChatGroq: {e}")
    st.stop()

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
     Use the provided context to generate a clear, concise, and accurate answer.
     Do not rely on any information outside of the context.
     Respond politely and concisely.

    <context>
    {context}
    <context>
    Question: {input}
    Answer:
    """
)

# Create the document processing chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Check if vectors are properly initialized
if st.session_state.vectors:
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
else:
    st.error("Failed to initialize vectors. Please check your connection and try again.")
    st.stop()

# User input
user_prompt = st.text_input("Input your prompt here")

if user_prompt:
    try:
        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed_time = time.process_time() - start_time
        st.write(f"Response time: {elapsed_time:.2f} seconds")

        # Display the response
        st.write(response.get('answer', "No answer found."))

        # Show related documents
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("---")
    except httpx.ConnectError as e:
        st.error(f"Connection error during query execution: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")



