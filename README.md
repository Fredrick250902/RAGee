## ⚡ RAGee
A Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs, index them into a local vector database (Endee), and query them using a large language model with semantic search.
This project uses:
- Endee for local vector storage
- Hugging Face embeddings (MiniLM)
- Groq-hosted LLaMA 3.3 (70B) for generation
- Streamlit for the UI
- Docker Compose for running Endee locally

### ✅ Why Endee?
Endee is chosen as the vector database for this project because it offers a rare combination of performance, flexibility, and simplicity, especially for local and self-hosted RAG applications.

While many vector databases focus on managed or cloud-first setups, Endee provides:
- High-performance ANN search using HNSW
- Explicit quantization control (INT8, INT16, FLOAT16, etc.) to balance speed, memory, and accuracy
- Native metadata and advanced filtering
- Hybrid search capabilities
- Persistent local storage via Docker volumes

Although some vector databases also support quantization, these features are often coupled with heavier infrastructure requirements or more complex configuration. Endee exposes quantization and other performance optimizations in a lightweight, developer-friendly manner, making it well suited for experimentation, local RAG workflows, and production-ready prototypes with minimal operational overhead.

### 🏗️ Project Structure
```text
rag-endee/
│
├── app.py                  # Streamlit RAG application
├── utils.py                # PDF processing + embedding utilities
├── docker-compose.yml      # Endee vector DB service
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
└── endee_data/             # Persistent vector storage (Docker volume)
```

### ⚙️ Prerequisites
Make sure you have the following installed:
- Python 3.9+
- Docker & Docker Compose
- Hugging Face API key
- Groq API key

### 🔐 Environment Variables
Create a .env file in the project root:
- HF_API_KEY = your_huggingface_api_key
- GROQ_API_KEY = your_groq_api_key


graph TD
    User([User]) --> UI[Streamlit UI]
    UI -->|PDF / Query| Embed[Hugging Face Embeddings]
    Embed -->|384d Vectors| DB[(Endee Vector DB)]
    DB -->|Top-k Context| LLM[Groq LLaMA 3.3 70B]
    LLM -->|Answer| UI
