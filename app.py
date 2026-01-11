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

# --- Translations ---
LANGUAGES = {
    "English": {
        "sidebar_title": "Configuration",
        "api_key_label": "Gemini API Key",
        "api_key_help": "Enter your Google Gemini API Key",
        "api_key_success": "API Key configured!",
        "api_key_error": "API Key error: {e}",
        "api_key_warning": "Please enter your Gemini API Key.",
        "url_error": "Error processing URL: {e}",
        "main_title": "📄 Tech Document RAG",
        "url_label": "Enter Document URL (e.g., https://docs.streamlit.io/)",
        "url_placeholder": "https://...",
        "analyze_button": "Analyze Document",
        "analyzing_spinner": "Analyzing document...",
        "analysis_success": "Document analyzed successfully! You can now ask questions.",
        "url_warning": "Please enter a URL.",
        "chat_placeholder": "Ask a question about the document...",
        "thinking_spinner": "Thinking...",
        "system_prompt": (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Keep the answer concise and use Markdown format. "
            "\n\n"
            "{context}"
        ),
        "error_occurred": "An error occurred: {e}",
        "language_label": "Language",
    },
    "日本語": {
        "sidebar_title": "設定",
        "api_key_label": "Gemini APIキー",
        "api_key_help": "Google Gemini APIキーを入力してください",
        "api_key_success": "APIキーが設定されました！",
        "api_key_error": "APIキーエラー: {e}",
        "api_key_warning": "Gemini APIキーを入力してください。",
        "url_error": "URL処理エラー: {e}",
        "main_title": "📄 技術ドキュメントRAG",
        "url_label": "ドキュメントのURLを入力してください (例: https://docs.streamlit.io/)",
        "url_placeholder": "https://...",
        "analyze_button": "ドキュメントを解析",
        "analyzing_spinner": "ドキュメントを解析中...",
        "analysis_success": "ドキュメントの解析が完了しました！質問を入力してください。",
        "url_warning": "URLを入力してください。",
        "chat_placeholder": "ドキュメントについて質問する...",
        "thinking_spinner": "考え中...",
        "system_prompt": (
            "あなたは質問回答タスクのアシスタントです。"
            "提供されたコンテキストのみを使用して回答してください。"
            "答えがわからない場合は、わからないと答えてください。"
            "回答は簡潔にまとめ、Markdown形式を使用してください。"
            "\n\n"
            "{context}"
        ),
        "error_occurred": "エラーが発生しました: {e}",
        "language_label": "言語",
    }
}

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "language" not in st.session_state:
    st.session_state.language = "English"

t = LANGUAGES[st.session_state.language]

# --- Sidebar ---
st.sidebar.title(t["sidebar_title"])

# Language selector
selected_lang = st.sidebar.selectbox(
    t["language_label"],
    options=list(LANGUAGES.keys()),
    index=list(LANGUAGES.keys()).index(st.session_state.language)
)
if selected_lang != st.session_state.language:
    st.session_state.language = selected_lang
    st.rerun()

api_key = st.sidebar.text_input(t["api_key_label"], type="password", help=t["api_key_help"])

if api_key:
    try:
        # Initialize client to verify key
        client = genai.Client(api_key=api_key)
        st.sidebar.success(t["api_key_success"])
    except Exception as e:
        st.sidebar.error(t["api_key_error"].format(e=e))
else:
    st.sidebar.warning(t["api_key_warning"])

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
        st.error(t["url_error"].format(e=e))
        return None

# --- Main UI ---
st.title(t["main_title"])

url_input = st.text_input(t["url_label"], placeholder=t["url_placeholder"])

if st.button(t["analyze_button"], disabled=not api_key):
    if url_input:
        with st.spinner(t["analyzing_spinner"]):
            vs = process_url(url_input)
            if vs:
                st.session_state.vector_store = vs
                st.session_state.messages = [] # Clear history for new doc
                st.success(t["analysis_success"])
    else:
        st.warning(t["url_warning"])

# --- Chat Interface ---
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input(t["chat_placeholder"], disabled=st.session_state.vector_store is None):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Logic
    with st.chat_message("assistant"):
        with st.spinner(t["thinking_spinner"]):
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
                
                # Create chain
                system_prompt = t["system_prompt"]
                
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
                st.error(t["error_occurred"].format(e=e))
