import os
import json
import shutil
import hashlib
from docx import Document
from dotenv import load_dotenv
from flask import Flask, render_template, request

from langchain_core.documents import Document as LCDocument
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from prompts import qa_system_prompt, contextualize_q_system_prompt


load_dotenv()


# Chat model using Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


app = Flask(__name__)


chat_history = []
conversation_store = {}

SCRAPED_DOCS_DIR = "scraping"
FAISS_PATH = "faiss_index"


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in conversation_store:
        conversation_store[session_id] = ChatMessageHistory()
    return conversation_store[session_id]


def get_document_loader():
    """Load scraped documents (.docx) from scraping/ directory."""
    os.makedirs(SCRAPED_DOCS_DIR, exist_ok=True)
    documents = []
    for f in sorted(os.listdir(SCRAPED_DOCS_DIR)):
        if f.lower().endswith(".docx") and not f.startswith("~$"):
            full_path = os.path.join(SCRAPED_DOCS_DIR, f)
            try:
                doc = Document(full_path)
                full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if full_text.strip():
                    documents.append(
                        LCDocument(
                            page_content=full_text,
                            metadata={"source": f, "title": f}
                        )
                    )
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


def _faiss_meta_path():
    return os.path.join(os.getcwd(), FAISS_PATH, "metadata.json")


def _current_docs_fingerprint():
    """Compute a SHA-256 fingerprint of all .docx files under scraping/."""
    if not os.path.isdir(SCRAPED_DOCS_DIR):
        return ""

    fp = hashlib.sha256()
    docx_files = [
        os.path.join(SCRAPED_DOCS_DIR, f)
        for f in sorted(os.listdir(SCRAPED_DOCS_DIR))
        if f.lower().endswith(".docx") and not f.startswith("~$")
    ]

    if not docx_files:
        return ""

    for full_path in docx_files:
        try:
            stat = os.stat(full_path)
            rel_path = os.path.relpath(full_path, SCRAPED_DOCS_DIR)
            fp.update(f"{rel_path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        except OSError:
            pass
    return fp.hexdigest()


def get_embeddings():
    """Load or rebuild the FAISS index only when scraped documents change."""
    index_path = os.path.join(os.getcwd(), FAISS_PATH)
    meta_path = _faiss_meta_path()
    current_fp = _current_docs_fingerprint()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # If an index exists and fingerprint matches, load it
    if os.path.isdir(index_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("fingerprint") == current_fp and current_fp != "":
                print("Loading existing FAISS index (fingerprint match)")
                return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Failed to load existing FAISS index: {e}")

    # Discover and load scraped .docx files
    documents = get_document_loader()
    if not documents:
        print("No .docx documents found in scraping/ to index.")
        if os.path.isdir(index_path) and os.path.exists(meta_path):
            try:
                return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Failed to load fallback index: {e}")
        return None

    # Rebuild the index safely in a staging directory
    print("Creating / updating FAISS index – Scraped documents changed")
    staging_path = index_path + "_staging"

    try:
        chunks = get_text_chunks(documents)
        if not chunks:
            print("No text chunks generated from documents.")
            return None

        # Build new index in memory
        new_db = FAISS.from_documents(chunks, embedding_model)

        if os.path.isdir(staging_path):
            shutil.rmtree(staging_path, ignore_errors=True)
        os.makedirs(staging_path, exist_ok=True)
        new_db.save_local(staging_path)

        with open(os.path.join(staging_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"fingerprint": current_fp}, f)

        if os.path.isdir(index_path):
            shutil.rmtree(index_path, ignore_errors=True)
        shutil.move(staging_path, index_path)

        print("FAISS index created/updated successfully.")
        return new_db

    except Exception as e:
        print(f"Error rebuilding FAISS index: {e}")
        if os.path.isdir(staging_path):
            shutil.rmtree(staging_path, ignore_errors=True)
        raise e


def get_retriever():
    db = get_embeddings()
    if db is None:
        raise RuntimeError("FAISS index is not available. Please ensure scraped .docx documents are placed in scraping/.")
    return db.as_retriever()


@app.route("/")
def index():
    return render_template("home.html")


def compute_confidence(query: str, retrieved_docs: list, answer_text: str, vector_store) -> dict:
    """Compute an explainable confidence estimate based on vector similarity scores."""
    if not retrieved_docs or vector_store is None:
        return {
            "score": None,
            "percent": None,
            "level": "Insufficient evidence",
            "explanation": "No relevant document chunks were retrieved from the scraped web documents."
        }

    try:
        docs_and_scores = vector_store.similarity_search_with_score(query, k=min(4, len(retrieved_docs)))
        if not docs_and_scores:
            return {
                "score": None,
                "percent": None,
                "level": "Insufficient evidence",
                "explanation": "Could not retrieve similarity scores for the context chunks."
            }

        # For unit-normalized embeddings, cosine similarity = 1.0 - (L2_distance^2 / 2.0)
        similarities = []
        for _, distance in docs_and_scores:
            cos_sim = max(0.0, min(1.0, 1.0 - (float(distance) / 2.0)))
            similarities.append(cos_sim)

        top_sim = similarities[0]
        avg_sim = sum(similarities) / len(similarities)

        combined = (0.7 * top_sim) + (0.3 * avg_sim)
        pct = int(round(combined * 100))

        answer_lower = answer_text.lower()
        unsupported_markers = [
            "do not contain", "does not contain", "cannot find", "not mentioned",
            "no information", "not provided", "insufficient evidence", "do not provide"
        ]
        if any(marker in answer_lower for marker in unsupported_markers):
            return {
                "score": round(combined, 3),
                "percent": min(pct, 30),
                "level": "Low confidence (insufficient evidence in documents)",
                "explanation": "The language model indicated that the scraped documents do not contain sufficient information to answer this question."
            }

        if combined >= 0.70:
            level = "High confidence"
        elif combined >= 0.50:
            level = "Moderate confidence"
        else:
            level = "Low confidence"

        explanation = (
            f"Calculated from semantic vector similarity ({pct}%) between your question "
            f"and the top retrieved scraped chunks (top chunk: {int(round(top_sim*100))}%, "
            f"average: {int(round(avg_sim*100))}%). "
            "Note: This is an estimated retrieval similarity metric, not a calibrated statistical probability."
        )

        return {
            "score": round(combined, 3),
            "percent": pct,
            "level": level,
            "explanation": explanation
        }

    except Exception as e:
        print(f"Error computing confidence score: {e}")
        return {
            "score": None,
            "percent": None,
            "level": "Estimated from retrieval",
            "explanation": "Similarity score calculation encountered an error."
        }


@app.route("/chat", methods=["GET", "POST"])
def document_display():
    if request.method == "GET":
        return render_template("chat.html", chat_history=chat_history)

    question = request.form.get("question", "").strip()
    if not question:
        return render_template("chat.html", chat_history=chat_history)

    try:
        retriever = get_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        response = conversational_rag_chain.invoke(
            {
                "input": question
            },
            config={
                "configurable": {
                    "session_id": "session_default"
                }
            }
        )

        answer_text = response.get("answer", "No response received.")
        retrieved_context = response.get("context", [])

        # Extract source metadata from retrieved document chunks
        sources_map = {}
        for doc in retrieved_context:
            raw_src = doc.metadata.get("source", "")
            fname = os.path.basename(raw_src) if raw_src else "Scraped Document"
            sources_map[fname] = sources_map.get(fname, 0) + 1

        sources_list = []
        for fname, count in sources_map.items():
            label = f"{count} passage{'s' if count > 1 else ''}"
            sources_list.append({
                "filename": fname,
                "pages_label": label
            })

        db = get_embeddings()
        confidence = compute_confidence(question, retrieved_context, answer_text, db)

        chat_history.append(question)
        chat_history.append({
            "answer": answer_text,
            "sources": sources_list,
            "confidence": confidence
        })

    except Exception as e:
        print(f"Error processing chat request: {e}")
        chat_history.append(question)
        chat_history.append({
            "answer": f"An error occurred while generating a response ({type(e).__name__}). Please check the server logs or try again shortly.",
            "sources": [],
            "confidence": {
                "score": 0,
                "percent": 0,
                "level": "Error",
                "explanation": "No response could be generated due to an error."
            }
        })

    return render_template(
        "chat.html",
        chat_history=chat_history
    )


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("chat.html", chat_history=chat_history), 500


if __name__ == "__main__":
    app.run(debug=True)
