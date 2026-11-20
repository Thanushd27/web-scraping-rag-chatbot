# Gen AI Web-Scraping RAG Chatbot

An intelligent Retrieval-Augmented Generation (RAG) conversational chatbot that automatically extracts knowledge from live websites, builds semantic vector indexes, and answers questions with conversational context, source citations, and confidence scoring.

---

## Architecture & How It Works

```
                        +----------------------------+
                        |  scraping/pages_name.txt   | (Target URLs)
                        +--------------+-------------+
                                       |
                                       v
                        +----------------------------+
                        |   Selenium + BeautifulSoup | (Web Scraper)
                        +--------------+-------------+
                                       | Clean Text (.docx)
                                       v
                        +----------------------------+
                        |    scraping/*.docx docs    |
                        +--------------+-------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      +-------------------------+             +-------------------------+
      |  HuggingFace Embeddings |             |   Flask Web UI (/chat)  |
      +------------+------------+             +------------+------------+
                   |                                       |
                   v                                       v
      +-------------------------+             +-------------------------+
      |    FAISS Vector Store   | <---------> |   Groq ChatGroq LLM     |
      |      (faiss_index/)     |  Retrieval  | (openai/gpt-oss-120b)   |
      +-------------------------+             +-------------------------+
```

1. **Web Scraping:** URLs listed in `scraping/pages_name.txt` are fetched via Selenium WebDriver (to render JavaScript-heavy content) and parsed with BeautifulSoup. The extracted text is cleaned using regular expressions and saved as Word documents (`.docx`) in `scraping/`.
2. **Text Chunking & Indexing:** `app.py` loads the `.docx` documents using `python-docx`, splits them into 1,000-character overlapping chunks with `RecursiveCharacterTextSplitter`, generates vector embeddings via HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, and builds a local FAISS index (`faiss_index/`).
3. **Smart Fingerprint Caching:** A SHA-256 fingerprint of the files in `scraping/` is stored in `faiss_index/metadata.json`. The index is loaded from cache on startup and only rebuilt if documents are added or updated.
4. **Conversational RAG Chain:** Uses LangChain's history-aware retriever to contextualize follow-up questions, queries the FAISS index for relevant chunks, and synthesizes answers via Groq LLM (`openai/gpt-oss-120b`) with source passage citations and confidence estimation.

---

## Tech Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Web Scraping** | Selenium, BeautifulSoup4, cloudscraper | Headless/browser-based HTML extraction and dynamic JS rendering |
| **Document Storage** | `python-docx` | Structured `.docx` files in `scraping/` |
| **Embeddings** | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` | Dense vector representations (384-dimensional) |
| **Vector Database** | FAISS (`faiss-cpu`) | High-performance similarity search and indexing |
| **LLM Inference** | Groq Cloud (`ChatGroq`) | Fast model inference (`openai/gpt-oss-120b` or Llama models) |
| **RAG Framework** | LangChain (`langchain`, `langchain-community`, `langchain-classic`) | History-aware retrieval, prompt templating, and runnable chains |
| **Web Application** | Flask | Modern, responsive chat UI with confidence metrics |

---

## Project Structure

```text
├── app.py                     # Main Flask web application and RAG pipeline
├── prompts.py                 # System prompts for QA and contextualization
├── requirements.txt           # Project Python dependencies
├── .env.example               # Template for environment variables
├── .gitignore                 # Git ignore rules
├── scraping/
│   ├── extract.ipynb          # Jupyter notebook for web scraping
│   ├── pages_name.txt         # List of target URLs to scrape
│   └── *.docx                 # Scraped website documents (knowledge base)
└── templates/
    ├── home.html              # Landing page
    ├── chat.html              # Interactive chat interface
    └── index.html             # Alternative query page
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- Google Chrome installed (for Selenium scraping)
- A free [Groq Cloud API Key](https://console.groq.com/keys)

### 2. Clone the Repository

```bash
git clone https://github.com/Thanushd27/web-scraping-rag-chatbot.git
cd web-scraping-rag-chatbot
```

### 3. Set Up Virtual Environment

On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Copy the template:
```bash
cp .env.example .env
```
*(On Windows PowerShell, use `Copy-Item .env.example .env`)*

Edit `.env` and insert your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Run the Application

```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```
On first launch, `app.py` automatically reads the documents in `scraping/`, builds the FAISS vector index, and launches the chat server.

---

## How to Re-Run Web Scraping

To scrape new pages or update the knowledge base:

1. Add your target URLs to `scraping/pages_name.txt` (one URL per line).
2. Open and run the scraper notebook:
   ```bash
   jupyter notebook scraping/extract.ipynb
   ```
3. The scraper will fetch the pages, clean the text, and write new `.docx` files to `scraping/`.
4. Restart `app.py`. The application will detect the new files via SHA-256 fingerprinting and automatically update the FAISS index.
