import logging
from config import Config
from langchain_community.vectorstores import Chroma

# Imports para Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Imports para LLM y QA Chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

logger = logging.getLogger(__name__)

def get_embeddings():
    """Retorna la instancia de embeddings configurada."""
    if Config.EMBEDDINGS_PROVIDER == "openai":
        return OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    else:
        return HuggingFaceEmbeddings(model_name=Config.HF_EMBEDDING_MODEL)

def get_llm():
    """Retorna la instancia del LLM configurado."""
    if Config.LLM_PROVIDER == "openai":
        logger.info(f"Usando ChatOpenAI con modelo: {Config.OPENAI_MODEL_NAME}")
        return ChatOpenAI(
            model=Config.OPENAI_MODEL_NAME, 
            temperature=0, 
            openai_api_key=Config.OPENAI_API_KEY
        )
    else:
        logger.info(f"Usando ChatOllama con modelo: {Config.LOCAL_LLM_MODEL}")
        return ChatOllama(model=Config.LOCAL_LLM_MODEL, temperature=0)

def setup_qa_chain():
    """Configura y retorna la cadena de recuperación y generación."""
    # 1. Cargar la base de datos vectorial
    embeddings = get_embeddings()
    logger.info(f"Cargando ChromaDB desde: {Config.CHROMA_DB_DIR}")
    vector_store = Chroma(
        persist_directory=Config.CHROMA_DB_DIR, 
        embedding_function=embeddings
    )
    
    # 2. Configurar el retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 3. Configurar el LLM
    llm = get_llm()
    
    # 4. Configurar el Prompt
    system_prompt = (
        "Eres un asistente útil que responde preguntas basándose estrictamente en los fragmentos de contexto proporcionados.\n"
        "Si no sabes la respuesta basándote en el contexto, simplemente di que no lo sabes. No inventes información.\n"
        "\n"
        "Contexto:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Ensamblar la cadena
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def query_system(question: str):
    """Ejecuta una consulta en el sistema y muestra la respuesta y fuentes."""
    rag_chain = setup_qa_chain()
    
    logger.info(f"Realizando consulta: '{question}'")
    response = rag_chain.invoke({"input": question})
    
    answer = response.get("answer")
    source_documents = response.get("context", [])
    
    print("\n" + "="*50)
    print(f"PREGUNTA: {question}")
    print("="*50)
    print(f"\nRESPUESTA:\n{answer}\n")
    print("-" * 50)
    print("FUENTES UTILIZADAS:")
    for i, doc in enumerate(source_documents):
        source = doc.metadata.get("source", "Desconocido")
        page = doc.metadata.get("page", "Desconocida")
        print(f"[{i+1}] Archivo: {source} | Página: {page}")
        # print(f"Extracto: {doc.page_content[:150]}...\n")
    print("="*50 + "\n")

if __name__ == "__main__":
    import sys
    # Permitir hacer preguntas por CLI
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
        query_system(user_question)
    else:
        # Ejemplo por defecto si no se pasa argumento
        ejemplo = "¿De qué tratan los documentos?"
        query_system(ejemplo)
