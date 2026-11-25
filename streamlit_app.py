"""
Interactive Web Application for LLM Pipeline.
Built with Streamlit and powered by Groq & local FastEmbed.
"""

import os
import streamlit as st
from pipeline import LLMPipeline
from config import get_credentials, get_openai_client, DEFAULT_CHAT_MODEL

# Configure Streamlit Page
st.set_page_config(
    page_title="LLM Pipeline & Python Tutorials",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { padding-left: 18px; padding-right: 18px; font-weight: 600; }
    .source-box { background: #f8f9fa; border-left: 4px solid #1E88E5; padding: 10px; margin-top: 10px; border-radius: 4px; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cached Pipeline Instance
@st.cache_resource
def load_pipeline():
    return LLMPipeline()

pipeline = load_pipeline()
api_key, base_url, model_name, provider = get_credentials()

# Sidebar
with st.sidebar:
    st.image("https://groq.com/wp-content/uploads/2024/03/Groq_Logo_Black.svg" if provider == "groq" else "https://openai.com/favicon.ico", width=140)
    st.title("Settings & Status")
    st.success(f"**Backend:** {provider.upper()}")
    st.info(f"**Model:** `{model_name}`")
    st.caption("Embeddings: `fastembed (bge-small)`")
    
    st.divider()
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    system_prompt = st.text_area(
        "System Prompt",
        value="You are an expert, encouraging Python tutor. Provide clear explanations and clean, idiomatic code examples.",
        height=100,
    )
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header
st.markdown('<div class="main-header">⚡ Python LLM Pipeline & Tutorials</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Chat with ultra-fast models, generate step-by-step coding tutorials, and query private documents via RAG.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Interactive Chat", "📖 Tutorial Generator", "🔍 Document Q&A (RAG)"])

# ----------------- TAB 1: CHAT -----------------
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Python LLM assistant. Ask me anything about Python, algorithms, or frameworks!"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question (e.g. 'How do Python context managers work?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            client = get_openai_client()
            chat_messages = [{"role": "system", "content": system_prompt}] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            try:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=chat_messages,
                    temperature=temperature,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error communicating with {provider.upper()}: {e}")

# ----------------- TAB 2: TUTORIAL GENERATOR -----------------
with tab2:
    st.subheader("Generate Structured Python Tutorial")
    st.write("Extract clean, structured curriculum plans with step-by-step code snippets using Pydantic.")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic_input = st.text_input("Tutorial Topic", value="FastAPI Dependency Injection with SQLite", placeholder="e.g. Python Decorators, Asyncio Streams")
    with col2:
        diff_level = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"], index=1)

    if st.button("Generate Tutorial Guide", type="primary"):
        if not topic_input.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner(f"Generating {diff_level} tutorial for '{topic_input}' on {model_name}..."):
                try:
                    tutorial = pipeline.generate_tutorial(topic_input, difficulty=diff_level)
                    
                    st.success("Tutorial Generated!")
                    st.markdown(f"## 📘 {tutorial.title}")
                    st.markdown(f"**Difficulty:** `{tutorial.difficulty}`")
                    st.markdown(f"**Overview:** {tutorial.summary}")
                    st.markdown(f"**Prerequisites:** {', '.join(tutorial.prerequisites)}")
                    st.divider()

                    for step in tutorial.steps:
                        with st.expander(f"Step {step.step_number}: {step.title}", expanded=True):
                            st.write(step.explanation)
                            st.code(step.code.strip(), language="python")
                except Exception as e:
                    st.error(f"Failed to generate tutorial: {e}")

# ----------------- TAB 3: DOCUMENT Q&A (RAG) -----------------
with tab3:
    st.subheader("Semantic Document Q&A (RAG)")
    st.write("Query private documents from `./data` using local vector embeddings and Groq reasoning.")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    existing_files = os.listdir(data_dir) if os.path.exists(data_dir) else []
    st.info(f"**Documents currently in knowledge base:** {', '.join(existing_files) if existing_files else 'None'}")

    # File upload
    uploaded_file = st.file_uploader("Add a new text/markdown file to knowledge base:", type=["txt", "md"])
    if uploaded_file is not None:
        save_path = os.path.join(data_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded and saved `{uploaded_file.name}`! Please reload index.")
        if st.button("Rebuild Index"):
            pipeline._rag_engine = None
            st.rerun()

    rag_query_input = st.text_input("Ask a question about your documents:", value="What is LlamaIndex?")
    if st.button("Search & Answer", type="primary"):
        if not rag_query_input.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant context & generating answer..."):
                try:
                    rag_response = pipeline.rag_query(rag_query_input)
                    st.markdown("### Answer")
                    st.markdown(rag_response)
                except Exception as e:
                    st.error(f"RAG query failed: {e}")
