import os
import logging
from config import Config
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Imports para Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

def get_embeddings():
    """Retorna la instancia de embeddings configurada."""
    if Config.EMBEDDINGS_PROVIDER == "openai":
        logger.info("Usando OpenAIEmbeddings")
        return OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    else:
        logger.info(f"Usando HuggingFaceEmbeddings con modelo: {Config.HF_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(model_name=Config.HF_EMBEDDING_MODEL)

def load_documents_robust(directory):
    """
    Carga todos los PDFs de un directorio de forma robusta.
    Ignora archivos corruptos y loguea errores sin romper el pipeline.
    """
    docs = []
    logger.info(f"Iniciando escaneo de PDFs en: {directory}")
    
    if not os.path.exists(directory):
        logger.error(f"El directorio no existe: {directory}")
        return docs

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                logger.info(f"Procesando: {file_path}")
                try:
                    loader = PyPDFLoader(file_path)
                    file_docs = loader.load()
                    # Opcional: Agregar más metadatos si fuera necesario
                    for doc in file_docs:
                        # Asegurarse de que el source sea claro
                        doc.metadata["source"] = file_path
                        doc.metadata["filename"] = file
                    docs.extend(file_docs)
                    logger.info(f"Éxito: {len(file_docs)} páginas extraídas de {file}")
                except Exception as e:
                    logger.error(f"Error procesando el archivo {file_path}. Se ignorará. Detalle: {e}")
                    
    logger.info(f"Total de páginas válidas cargadas: {len(docs)}")
    return docs

def ingest_data():
    """Pipeline principal de ingesta: Carga, splitea y persiste."""
    # 1. Cargar documentos
    documents = load_documents_robust(Config.PDF_DIRECTORY)
    
    if not documents:
        logger.warning("No se encontraron documentos válidos para procesar. Abortando ingesta.")
        return

    # 2. Chunking (dividir el texto)
    logger.info("Iniciando partición de texto (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Se generaron {len(chunks)} chunks de texto.")

    # 3. Inicializar modelo de Embeddings
    embeddings = get_embeddings()

    # 4. Crear y persistir el Vector Store (Chroma)
    logger.info(f"Guardando embeddings en ChromaDB en: {Config.CHROMA_DB_DIR}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=Config.CHROMA_DB_DIR
    )
    
    # En versiones recientes de Chroma, se guarda automáticamente en disco
    # pero forzar persist() asegura la retrocompatibilidad en ciertas versiones.
    # vector_store.persist() ya no es estrictamente necesario en Chroma > 0.4.x,
    # el persist_directory basta.
    
    logger.info("¡Ingesta completada exitosamente!")

if __name__ == "__main__":
    ingest_data()
