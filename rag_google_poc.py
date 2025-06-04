# rag_google_poc.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate # Added PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever # Added MultiQueryRetriever


# --- 0. Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's in your .env file.")

# --- 1. Load Documents ---
DATA_PATH = "data"
print(f"Looking for documents in: {os.path.abspath(DATA_PATH)}")

loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=False)
documents = loader.load()

if not documents:
    print(f"No documents found in '{DATA_PATH}'. Make sure you have PDF/TXT files there.")
    exit()
print(f"Loaded {len(documents)} document(s)/page(s).")

# --- 2. Split Documents into Chunks ---
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

if not chunks:
    print("No chunks were created. Check document loading and splitting.")
    exit()
print(f"Split into {len(chunks)} chunks.")

# --- 3. Initialize Embedding Model (Using Google AI) ---
print("Initializing Google AI embedding model...")
# Common model for embeddings via Gemini API. Check Google AI Studio for latest recommended embedding models.
# Other options might include "models/text-embedding-004" or specific Gemma embedding models if listed.
embeddings_model_name = "models/embedding-001" # A commonly available and effective model
try:
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=embeddings_model_name,
        google_api_key=GOOGLE_API_KEY
        # task_type="retrieval_document" # Optional: can sometimes improve relevance for RAG
    )
    print(f"Google AI embedding model '{embeddings_model_name}' initialized.")
except Exception as e:
    print(f"Error initializing Google AI Embeddings: {e}")
    print("Consider falling back to local HuggingFace embeddings if Google API access is an issue.")
    # Fallback to local HuggingFace Embeddings (uncomment if needed)
    # print("Initializing local HuggingFace embedding model as fallback...")
    # embeddings_model_name_hf = "all-MiniLM-L6-v2"
    # embeddings_model = HuggingFaceEmbeddings(
    #     model_name=embeddings_model_name_hf,
    #     model_kwargs={'device': 'cpu'}
    # )
    # print(f"Local HuggingFace embedding model '{embeddings_model_name_hf}' loaded.")
    # exit() # Or handle differently

# --- 4. Create Vector Store and Embed Documents ---
# vectorstore_directory = "vectorstore_db_google"
# print(f"Creating/loading vector store at '{os.path.abspath(vectorstore_directory)}'...")

# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings_model,
#     persist_directory=vectorstore_directory
# )
# print("Vector store created and documents embedded using Google AI embeddings.")

# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={'k': 3}
# )
# print("Retriever created.")

vectorstore_directory = "vectorstore_db_google"
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings_model, persist_directory=vectorstore_directory)
print("Vector store created/loaded.")

# Define the base retriever
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3} # k for each generated query
)

# --- 5. Initialize LLM (Using Google AI) ---
print("Initializing Google AI LLM...")

# Option 1: Using a specific Gemma model if directly available and known to be free
# model_name_llm = "gemma-3-27b-it" # Check Google AI documentation for exact available model names
model_name_llm = "gemma-3-12b-it" # A common Gemma instruction-tuned model

# Option 2: Using a general high-performing Gemini model (often has good free tier)
# model_name_llm = "gemini-1.5-flash-8b" # Recommended for good balance of performance and cost/free tier access

try:
    llm = ChatGoogleGenerativeAI(
        model=model_name_llm,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        # convert_system_message_to_human=True # Sometimes useful for compatibility
    )
    print(f"Google AI LLM initialized with model '{model_name_llm}'.")
except Exception as e:
    print(f"Error initializing Google LLM: {e}")
    print("Please ensure the model name is correct and your API key has access.")
    exit()

# --- NEW: Setup MultiQueryRetriever ---
print("Setting up MultiQueryRetriever...")
# Using the default prompt for generating queries, which is usually quite good.
# The LLM (Gemma 3 27b / Gemini 1.5 Flash) will be used to generate these queries.
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
print("MultiQueryRetriever is now set up.")

# --- 6. Define the RAG Chain ---
print("Defining RAG chain...")

# The system instruction might need slight adjustment based on how ChatGoogleGenerativeAI handles system prompts.
# Often, for Gemini models, putting instructions directly in the user part of the prompt with context works well.
template = """
You are an assistant for question-answering tasks based on union agreements.
Use ONLY the following pieces of retrieved context to answer the question.
If you don't know the answer from the context provided, just say that you don't know.
Do not make up an answer or use outside knowledge.
Keep the answer concise and directly related to the provided context.
Include the headings and subheadings and sections in the answer.
Include the quoted text from the context in the answer.

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
print("RAG chain defined.")

# --- 7. Test the RAG Chain (Simple CLI) ---
if __name__ == "__main__":
    print("\n--- Union Agreement Chatbot PoC (Google AI - Windows) ---")
    print(f"Using LLM: {model_name_llm}, Embeddings: {embeddings_model_name}")
    print("Type 'exit' to quit.")
    while True:
        user_question = input("\nAsk a question about the union agreement: ")
        if user_question.lower() == 'exit':
            break
        if not user_question.strip():
            continue

        print(f"Processing with Google AI (Model: {model_name_llm})...")
        try:
            # Optional: Uncomment to see retrieved documents
            # retrieved_docs = retriever.invoke(user_question)
            # print("\n--- Retrieved Documents ---")
            # for i, doc in enumerate(retrieved_docs):
            #     print(f"Doc {i+1} (source: {doc.metadata.get('source', 'N/A')} page: {doc.metadata.get('page', 'N/A')}):\n{doc.page_content[:300]}...\n---")

            answer = rag_chain.invoke(user_question)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"An error occurred: {e}")
            # import traceback
            # traceback.print_exc()