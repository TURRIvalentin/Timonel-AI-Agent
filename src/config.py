import os
import logging
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde el archivo .env si existe
load_dotenv()

class Config:
    """Clase para centralizar la configuración del sistema."""
    
    # Rutas
    PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", r"C:\Users\valen\OneDrive\Escritorio\Timonel")
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    
    # Proveedores
    EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "huggingface").lower()
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # Modelos
    HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    @classmethod
    def validate(cls):
        """Valida que las configuraciones necesarias existan."""
        if not os.path.exists(cls.PDF_DIRECTORY):
            logger.warning(f"El directorio de PDFs no existe: {cls.PDF_DIRECTORY}")
            
        if cls.EMBEDDINGS_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY es requerida cuando EMBEDDINGS_PROVIDER es 'openai'")
            
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY es requerida cuando LLM_PROVIDER es 'openai'")

        logger.info("Configuración cargada y validada correctamente.")

# Al importar, se validan los parámetros base
Config.validate()
