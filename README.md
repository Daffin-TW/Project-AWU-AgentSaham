# 🤖 AWU Stock AI Agent

> **AI-powered stock market assistant** that scrapes financial news, embeds it into a local vector database, and answers investor questions using Retrieval-Augmented Generation (RAG).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**AWU Stock AI Agent** is an end-to-end intelligent assistant for Indonesian stock and crypto market analysis. It combines **automated web scraping**, **semantic search with embeddings**, and a **conversational AI agent** to deliver context-aware, news-grounded financial insights — all through an interactive Streamlit chat interface.

> ⚠️ **Disclaimer:** This application is for **informational purposes only** and does not constitute financial advice. Always consult a licensed financial advisor before making investment decisions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📰 **News Scraper** | Crawls and scrapes investment news articles from [Kontan.co.id](https://www.kontan.co.id) for a configurable date range |
| 🧠 **RAG Pipeline** | Embeds scraped articles into a local **ChromaDB** vector store for semantic retrieval |
| 💬 **AI Chat Agent** | Conversational agent powered by **OpenAI o4-mini** with full chat history awareness |
| 🔍 **Semantic Search** | Uses **Qwen3-Embedding-0.6B** (via HuggingFace) to find the most relevant news for any query |
| 📝 **Article Summarizer** | Summarizes retrieved documents using a fine-tuned Indonesian T5 model |
| 🖥️ **Streamlit UI** | Clean, responsive web interface with sidebar configuration controls |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (main.py)                   │
│              Sidebar: Config & Scraper  │  Chat Interface       │
└────────────────────┬────────────────────┬───────────────────────┘
                     │                    │
          ┌──────────▼──────┐   ┌─────────▼──────────────┐
          │   WebScraper    │   │        AIAgent          │
          │ (news_scrapper) │   │      (llm_agent)        │
          └──────────┬──────┘   └──────┬──────────────────┘
                     │                 │
          ┌──────────▼──────┐   ┌──────▼──────────────────┐
          │  CSV Dataset    │   │    RAG Chain (LangChain) │
          │  (dataset/)     │   │  History-aware Retriever │
          └─────────────────┘   └──────┬──────────────────┘
                                       │
                     ┌─────────────────▼──────────────────┐
                     │         ChromaEmbeddings            │
                     │  Qwen3-Embedding-0.6B (HuggingFace) │
                     │         ChromaDB (local)            │
                     └────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Project-AWU-AgentSaham/
├── main.py                   # Streamlit app entry point
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
├── .gitignore
├── LICENSE
│
├── module/
│   ├── __init__.py           # Package exports
│   ├── news_scrapper.py      # WebScraper — crawls Kontan.co.id
│   ├── embedding.py          # ChromaEmbeddings — vector store management
│   ├── summarizer.py         # ArticleSummarizer — Indonesian T5 model
│   └── llm_agent.py          # AIAgent — RAG-based conversational agent
│
├── dataset/                  # Scraped CSV files (auto-generated)
│   └── YYYY-MM-DD_<N>p.csv
│
├── chroma_db/                # Persisted ChromaDB vector store (auto-generated)
│
└── .streamlit/
    └── config.toml           # Streamlit theme configuration
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10** or higher
- An **OpenAI API key**
- GPU recommended for local embedding inference (CUDA-compatible)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Project-AWU-AgentSaham.git
cd Project-AWU-AgentSaham
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 5. Run the Application

```bash
streamlit run main.py
```

The app will be available at `http://localhost:8501`.

---

## 🔧 Usage Guide

### Step 1 — Scrape News Data

In the **sidebar**, set a start date and scraping period (1–90 days), then click **🔍 Mulai Scraping**. The app will crawl investment news from Kontan.co.id and save the results as a CSV file in the `dataset/` directory.

### Step 2 — Build the Vector Store

Run the embedding module once to index your scraped dataset into ChromaDB:

```bash
python -m module.embedding
```

This only needs to be done once (or when you add new data). The vector store is persisted locally in `chroma_db/`.

### Step 3 — Chat with the Agent

Use the chat interface to ask questions about stocks, crypto, or market trends. The agent will retrieve the most relevant news articles and generate a grounded, source-cited response.

**Example questions:**
- *"Bagaimana pergerakan saham BBRI minggu ini?"*
- *"Apa dampak kebijakan Fed terbaru terhadap pasar kripto?"*
- *"Analisis sentimen berita seputar saham TLKM?"*

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **UI Framework** | [Streamlit](https://streamlit.io/) |
| **LLM** | OpenAI `o4-mini-2025-04-16` |
| **Agent Framework** | [LangChain](https://www.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/) |
| **Embedding Model** | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (HuggingFace) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) |
| **Summarizer** | [thivh/t5-base-indonesian-summarization](https://huggingface.co/thivh/t5-base-indonesian-summarization-cased-finetuned-indosum) |
| **Web Scraping** | [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) + Requests |
| **Data Source** | [Kontan.co.id](https://www.kontan.co.id) — Investment News |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/<your-username>/Project-AWU-AgentSaham/issues).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ as part of the **MBKM — CV Advance with Unity** program

</div>
