# Python LLM Pipeline & Interactive Web Application

A high-performance Python LLM pipeline and interactive browser app powered by **Groq** (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`) and local **FastEmbed** embeddings, with full RAG document Q&A and Pydantic structured tutorial generation.

---

## 🌐 Live Web UI (Streamlit)

Your application is currently running locally at:
👉 **[http://localhost:8501](http://localhost:8501)**

### Web Features:
1. **💬 Real-Time Chat:** Multi-turn conversational interface with streaming tokens and customizable system prompt/temperature.
2. **📖 Structured Tutorial Generator:** Enter any programming topic (e.g. *FastAPI Background Tasks*, *Asyncio*) to receive a structured curriculum with code blocks and explanations.
3. **🔍 Document Q&A (RAG):** Upload `.txt` or `.md` files directly in your browser and ask questions against your private knowledge base using local vector embeddings.

---

## 🚀 How to Deploy the Web App Online (100% Free)

### Option A: Streamlit Community Cloud (Recommended)
1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Deploy LLM Pipeline"
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and click **New App**.
3. Select your repository and set the main file path to:
   ```text
   streamlit_app.py
   ```
4. Click **Advanced Settings** -> **Secrets**, and paste your API key:
   ```toml
   GROQ_API_KEY = "gsk_..."
   CHAT_MODEL = "openai/gpt-oss-120b"
   ```
5. Click **Deploy!** Your web app will receive a public URL (e.g., `https://your-app.streamlit.app`) that anyone can use in their browser.

---

### Option B: Deploy with Docker
```bash
docker build -t llm-web-app .
docker run -p 8501:8501 --env-file .env llm-web-app streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 📁 Repository Structure

```text
Pipeline/
├── streamlit_app.py                  # Streamlit Web UI (Chat, Tutorials, RAG)
├── app.py                            # FastAPI REST API server
├── pipeline.py                       # Core LLM pipeline engine
├── config.py                         # Environment & Groq credentials manager
├── requirements.txt                  # Dependencies
├── Dockerfile                        # Container specification
├── data/                             # Knowledge base files for RAG
│   └── llamaindex.txt
└── tutorials/                        # 6 Step-by-step modular scripts
    ├── 01_basic_completion.py
    ├── 02_streaming_responses.py
    ├── 03_structured_output.py
    ├── 04_embeddings_and_search.py
    ├── 05_rag_pipeline_llamaindex.py
    └── 06_chat_memory_pipeline.py
```
