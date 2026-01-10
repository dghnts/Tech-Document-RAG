import streamlit as st
from google import genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import bs4
import os

# Set User-Agent for LangChain
os.environ["USER_AGENT"] = "TechDocRAG/1.0"

# Required libraries:
# streamlit
# google-genai
# langchain
# langchain-community
# langchain-huggingface
# langchain-google-genai
# faiss-cpu
# sentence-transformers
# beautifulsoup4

# --- Page Configuration ---
st.set_page_config(page_title="Tech Doc RAG", layout="wide")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar ---
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key")

if api_key:
    try:
        # Initialize client to verify key
        client = genai.Client(api_key=api_key)
        st.sidebar.success("API Key configured!")
    except Exception as e:
        st.sidebar.error(f"API Key error: {e}")
else:
    st.sidebar.warning("Please enter your Gemini API Key.")

# --- Functions ---
def process_url(url):
    try:
        # Load document
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    ["main", "article", "div", "h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]
                )
            ),
        )
        docs = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        
        # Vector Store
        vector_store = FAISS.from_documents(splits, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return None

# --- Main UI ---
st.title("📄 Tech Document RAG")

url_input = st.text_input("Enter Document URL (e.g., https://docs.streamlit.io/)", placeholder="https://...")

if st.button("Analyze Document", disabled=not api_key):
    if url_input:
        with st.spinner("Analyzing document..."):
            vs = process_url(url_input)
            if vs:
                st.session_state.vector_store = vs
                st.session_state.messages = [] # Clear history for new doc
                st.success("Document analyzed successfully! You can now ask questions.")
    else:
        st.warning("Please enter a URL.")

# --- Chat Interface ---
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document...", disabled=st.session_state.vector_store is None):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Logic
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                
                # Create chain
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, say that you don't know. "
                    "Keep the answer concise and use Markdown format. "
                    "\n\n"
                    "{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")
