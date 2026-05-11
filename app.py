import streamlit as st
import os
import sys

# --- FIX PARA CHROMADB EN STREAMLIT CLOUD ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass
# --------------------------------------------


# Agregar src al path para poder importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import Config
from query import setup_qa_chain

st.set_page_config(page_title="Timonel RAG", page_icon="📘", layout="wide")

st.title("📘 Timonel: Buscador Inteligente de Documentos")
st.markdown("Haz preguntas y obtén respuestas basadas en tus PDFs locales.")

# Sidebar con información del sistema
with st.sidebar:
    st.header("Configuración Actual")
    st.write(f"**Directorio PDFs:** `{Config.PDF_DIRECTORY}`")
    st.write(f"**Vector Store:** `{Config.CHROMA_DB_DIR}`")
    st.write(f"**Embeddings:** `{Config.EMBEDDINGS_PROVIDER}`")
    st.write(f"**LLM:** `{Config.LLM_PROVIDER}`")
    
    st.markdown("---")
    st.markdown("""
    **Instrucciones:**
    1. Asegúrate de haber ejecutado `python src/ingest.py` primero para indexar tus PDFs.
    2. Escribe tu pregunta en la barra inferior.
    3. El sistema buscará en la base de datos y generará una respuesta basada estrictamente en los documentos.
    """)

# Inicializar QA chain en session state para no recargar en cada interacción
if "qa_chain" not in st.session_state:
    with st.spinner("Inicializando sistema RAG y conectando a ChromaDB..."):
        try:
            st.session_state.qa_chain = setup_qa_chain()
        except Exception as e:
            st.error(f"Error al inicializar el sistema: {e}")
            st.stop()

# Manejo del historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Ver fuentes documentales"):
                for i, doc in enumerate(msg["sources"]):
                    source = doc.metadata.get("source", "Desconocido")
                    page = doc.metadata.get("page", "Desconocida")
                    st.markdown(f"**Fuente {i+1}:** Archivo `{source}` | **Página:** `{page}`")
                    st.caption(f'"{doc.page_content[:300]}..."')

# Input del usuario
if question := st.chat_input("Escribe tu pregunta sobre los documentos..."):
    # Agregar pregunta al historial
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos y generando respuesta..."):
            try:
                response = st.session_state.qa_chain.invoke({"input": question})
                answer = response.get("answer")
                sources = response.get("context", [])
                
                st.markdown(answer)
                
                # Mostrar fuentes
                if sources:
                    with st.expander("Ver fuentes documentales"):
                        for i, doc in enumerate(sources):
                            source = doc.metadata.get("source", "Desconocido")
                            page = doc.metadata.get("page", "Desconocida")
                            st.markdown(f"**Fuente {i+1}:** Archivo `{source}` | **Página:** `{page}`")
                            st.caption(f'"{doc.page_content[:300]}..."')
                
                # Guardar en historial
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la consulta: {e}")
