import streamlit as st
import os
# from dotenv import load_dotenv
import glob

# --- Your existing RAG imports ---
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate #, PromptTemplate # Not needed if using default MultiQuery prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- 0. Load Environment Variables ---
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

if not GOOGLE_API_KEY:
    # For Streamlit Cloud, you'd set secrets there, not via .env
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("GOOGLE_API_KEY not found. Please set it in your environment or Streamlit secrets.")
        st.stop()

# --- Global variable for the RAG chain (to load models only once) ---
# We'll initialize this in a function with st.cache_resource
rag_chain_global = None
model_name_llm = "gemma-3-12b-it"

# --- Function to Initialize and Load RAG Components (Cached) ---
@st.cache_resource # This decorator caches the resource across reruns and sessions
def load_rag_pipeline():
    # st.write("Initializing RAG pipeline (this may take a moment on first run)...")

    # 1. Load Documents (Simplified for example - consider how to manage this in a deployed app)
    # For a real app, you might have a fixed set of documents or a way to upload them.
    # For this example, let's assume documents are loaded and processed once.
    # If documents change, the cache needs to be cleared or managed.
    data_path = "data"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        st.error(f"Data directory '{data_path}' is empty or not found. Please add your union agreement documents.")
        st.stop()
        
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=False)
    documents = loader.load()
    if not documents:
        st.error("No documents successfully loaded from the 'data' directory.")
        st.stop()

    # 2. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        st.error("Document splitting resulted in no chunks.")
        st.stop()

    # 3. Initialize Embedding Model
    embeddings_model_name = "models/embedding-001"
    embeddings_model = GoogleGenerativeAIEmbeddings(model=embeddings_model_name, google_api_key=GOOGLE_API_KEY)

    # 4. Create Vector Store and Base Retriever
    # For a deployed app, you might persist the vector store more robustly or build it on startup.
    # Using a persisted directory for Chroma can save re-computation if data doesn't change.
    vectorstore_directory = "vectorstore_db_google_streamlit" # Use a different dir if needed
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings_model, persist_directory=vectorstore_directory)
    
    base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 3})

    # 5. Initialize LLM
    # model_name_llm = "gemma-3-12b-it" # A common Gemma instruction-tuned model
    # model_name_llm = "gemini-1.5-flash-latest" # Or your "gemma-3-27b-it" or other Gemma variant
    llm = ChatGoogleGenerativeAI(model=model_name_llm, google_api_key=GOOGLE_API_KEY, temperature=0.2)

    # 6. Setup MultiQueryRetriever
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    # 7. Define the RAG Chain Prompt
    template = """
You are an assistant for question-answering tasks based on union agreements.
Use ONLY the following pieces of retrieved context to answer the question.
If you don't know the answer from the context provided, just say that you don't know.
Do not make up an answer or use outside knowledge.
Keep the answer concise and directly related to the provided context.
Include the headings and subheadings and sections in the answer if present in the context and relevant.
Include the quoted text from the context in the answer if it directly supports the answer.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n---\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # st.success("RAG pipeline initialized successfully!")
    return rag_chain

# --- Load the RAG chain (will run once and be cached) ---
# Ensure GOOGLE_API_KEY is available before calling this
if GOOGLE_API_KEY:
    rag_chain_global = load_rag_pipeline()
else:
    st.info("API Key not found. RAG pipeline not loaded.")


# --- Streamlit App UI ---
# Add PDF download links in the sidebar
pdf_files = glob.glob(os.path.join("data", "*.pdf"))
if pdf_files:
    st.sidebar.header("Union Agreement PDFs")
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        with open(pdf_path, "rb") as f:
            st.sidebar.download_button(
                label=f"Download {pdf_name}",
                data=f,
                file_name=pdf_name,
                mime="application/pdf",
                type="primary"
            )
else:
    st.sidebar.info("No PDF agreements found in the data folder.")

# --- Sticky Header Title ---
st.markdown(
    """
    <style>
    .sticky-title {
        position: fixed;
        top: 0;
        z-index: 1000;
        margin: 3.75rem 0 0 0;
        text-align: center;
        background: #0E1117;
        border-radius: .5em;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > section.stMain.st-emotion-cache-bm2z3a.en45cdb1 > div.stMainBlockContainer.block-container.st-emotion-cache-1cei9z1.en45cdb4 > div > div > div.stElementContainer.element-container.st-emotion-cache-17lr0tt.e1lln2w81 {
        margin-top: 3rem;
        
    }

    </style>
    <div class="sticky-title">
        <h1>📜 MEEA Union Agreement Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the union agreement:"):
    if not rag_chain_global:
        st.error("RAG Pipeline is not available. Please check API key and configurations.")
    else:
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    def stream_response():
                        full_response = ""
                        for chunk in rag_chain_global.stream(prompt):
                            full_response += chunk
                            yield chunk
                        return full_response
                    # Stream the response and capture the full output
                    full_response = st.write_stream(stream_response())
                    # After streaming, append to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    # Optionally add error to chat history
                    # st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})