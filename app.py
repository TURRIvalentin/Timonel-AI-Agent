"""Timonel RAG — Streamlit web interface."""
from __future__ import annotations

import sys

# ChromaDB requires sqlite3 >= 3.35. On Streamlit Cloud the system sqlite3 is
# too old; pysqlite3-binary ships a compatible version as a drop-in replacement.
try:
    import pysqlite3  # type: ignore[import-untyped]

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import streamlit as st

from src.config import settings
from src.core.retrieval import setup_qa_chain
from src.db.repository import get_recent_logs, save_query_log
from src.db.session import init_db

init_db()

st.set_page_config(page_title="Timonel RAG", page_icon="📘", layout="wide")
st.title("📘 Timonel: Intelligent Document Search")
st.markdown("Ask questions and get answers grounded in your local PDF library.")

with st.sidebar:
    st.header("Current Configuration")
    st.write(f"**PDF directory:** `{settings.pdf_directory}`")
    st.write(f"**Vector store:** `{settings.chroma_db_dir}`")
    st.write(f"**Embeddings:** `{settings.embeddings_provider}`")
    st.write(f"**LLM:** `{settings.llm_provider}`")

    st.markdown("---")
    st.markdown(
        """
        **Instructions:**
        1. Run `python cli.py ingest` to index your PDFs.
        2. Type your question in the input box below.
        3. The system retrieves relevant passages and generates a grounded answer.
        """
    )

    st.markdown("---")
    st.header("📜 Recent Queries")
    recent_logs = get_recent_logs(limit=5)
    if recent_logs:
        for log in recent_logs:
            with st.expander(
                f"🕒 {log.timestamp.strftime('%H:%M:%S')} — {log.question[:30]}…"
            ):
                st.write(f"**Q:** {log.question}")
                st.write(f"**A:** {log.answer[:200]}…")
                st.caption(f"Sources: {log.sources}")
    else:
        st.info("No queries recorded yet.")

if "qa_chain" not in st.session_state:
    with st.spinner("Initialising RAG system…"):
        try:
            st.session_state.qa_chain = setup_qa_chain(settings)
        except Exception as exc:
            st.error(f"Failed to initialise the RAG system: {exc}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View source passages"):
                for i, doc in enumerate(msg["sources"]):
                    src = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i + 1}:** `{src}` — page `{page}`")
                    st.caption(f'"{doc.page_content[:300]}…"')

if question := st.chat_input("Ask a question about the documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                response = st.session_state.qa_chain.invoke({"input": question})
                answer = response.get("answer", "")
                sources = response.get("context", [])

                st.markdown(answer)

                if sources:
                    with st.expander("View source passages"):
                        for i, doc in enumerate(sources):
                            src = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")
                            st.markdown(f"**Source {i + 1}:** `{src}` — page `{page}`")
                            st.caption(f'"{doc.page_content[:300]}…"')

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
                save_query_log(question, answer, sources)
            except Exception as exc:
                st.error(f"Error processing query: {exc}")
