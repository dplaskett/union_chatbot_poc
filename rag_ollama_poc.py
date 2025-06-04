# rag_ollama_poc.py
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # For local embeddings
from langchain_community.llms import Ollama # For Ollama LLM
from langchain_community.chat_models import ChatOllama # Alternative for chat-tuned Ollama models
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Load Documents ---
DATA_PATH = "data"
print(f"Looking for documents in: {os.path.abspath(DATA_PATH)}")

loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=False)
# For .txt files:
# loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
documents = loader.load()

if not documents:
    print(f"No documents found in '{DATA_PATH}'. Make sure you have PDF/TXT files there.")
    exit()

print(f"Loaded {len(documents)} document(s)/page(s).")
# print(f"First document content snippet: {documents[0].page_content[:200]}")

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
# print(f"First chunk content: {chunks[0].page_content}")

# --- 3. Initialize Local Embedding Model ---
print("Initializing local embedding model (this might download the model on first run)...")
# Using a popular, lightweight, and effective sentence transformer model
# You can choose other models from Hugging Face: https://huggingface.co/models?library=sentence-transformers
embeddings_model_name = "all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_name,
    model_kwargs={'device': 'cpu'} # Explicitly use CPU if no compatible GPU or to ensure CPU usage
)
print(f"Local embedding model '{embeddings_model_name}' loaded.")

# --- 4. Create Vector Store and Embed Documents ---
vectorstore_directory = "vectorstore_db_ollama"
print(f"Creating/loading vector store at '{os.path.abspath(vectorstore_directory)}'...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=vectorstore_directory
)
print("Vector store created and documents embedded using local embeddings.")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3}
)
print("Retriever created.")

# --- 5. Initialize Local LLM (Ollama) ---
print("Initializing Ollama LLM...")
# Ensure Ollama application is running and the model is pulled (e.g., 'ollama pull llama3:8b')
# You can use ChatOllama for models fine-tuned for chat, or Ollama for base models
llm = ChatOllama(
    model="llama3.2:3b", # Replace with your pulled model, e.g., "mistral", "phi3"
    temperature=0.2
)
# Alternatively, for non-chat specific models or simpler completion:
# llm = Ollama(model="llama3.1:8b", temperature=0.2)

print(f"Ollama LLM initialized with model 'llama3.2:3b'.") # Or your chosen model

# --- 6. Define the RAG Chain ---
print("Defining RAG chain...")

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
    print("\n--- Union Agreement Chatbot PoC (Ollama Local LLM - Windows) ---")
    print("Ensure Ollama is running with your chosen model.")
    print("Type 'exit' to quit.")
    while True:
        user_question = input("\nAsk a question about the union agreement: ")
        if user_question.lower() == 'exit':
            break
        if not user_question.strip():
            continue

        print("Processing with local LLM...")
        try:
            # Optional: Uncomment to see retrieved documents for debugging
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
            # traceback.print_exc() # For more detailed error info