# DocuQuery AI - Powered by Groq

DocuQuery AI is an intelligent document query application that allows users to upload PDF or TXT files and ask questions related to the content. The application uses advanced embeddings, vector search, and large language models (LLMs) to retrieve relevant information and provide concise, context-aware answers. Powered by Groq and LangChain, DocuQuery AI delivers fast and accurate document querying.

## Features

- **Document Upload:** Supports PDF and TXT file uploads for content processing.
- **Smart Document Splitting:** Breaks down large documents into manageable chunks for efficient processing.
- **Vector Search:** Uses FAISS for fast document similarity search.
- **Context-Aware Responses:** Provides accurate answers based on uploaded content using Groq's LLM.
- **Interactive UI:** Simple and user-friendly interface built with Streamlit.

## Technologies Used

- **Python 3.9+**
- **Streamlit:** For the user interface.
- **Groq API:** For leveraging large language models.
- **LangChain:** For document processing and prompt management.
- **FAISS:** For efficient vector storage and similarity search.
- **HuggingFace Embeddings:** For embedding document text.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vinaymh2025/DocuQueryAI-usingGroq.git
   cd docuquery-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the web app in your browser (usually at `http://localhost:8501`).
2. Upload a PDF or TXT file.
3. Enter your query in the input box.
4. View the generated response and related document context.

---

*Happy querying with DocuQuery AI!*

