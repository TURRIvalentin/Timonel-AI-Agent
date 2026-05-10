import os
import logging
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde el archivo .env si existe (uso local)
load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """
    Lee un valor de configuración con este orden de prioridad:
    1. Variable de entorno del sistema (incluye Streamlit Cloud secrets que se inyectan como env vars)
    2. st.secrets (acceso directo al store de Streamlit Cloud)
    3. Valor por defecto
    """
    # 1. Intentar desde variables de entorno (funciona en local con .env y en Streamlit Cloud)
    value = os.getenv(key, "")
    if value:
        return value

    # 2. Intentar desde st.secrets (fallback explícito para Streamlit Cloud)
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass  # Fuera de un contexto de Streamlit, ignorar

    return default


class Config:
    """Clase para centralizar la configuración del sistema."""

    @classmethod
    def _load(cls):
        """Carga todos los valores de configuración dinámicamente."""
        # Rutas
        cls.PDF_DIRECTORY = _get_secret("PDF_DIRECTORY", "./data")
        cls.CHROMA_DB_DIR = _get_secret("CHROMA_DB_DIR", "./chroma_db")

        # Proveedores
        cls.EMBEDDINGS_PROVIDER = _get_secret("EMBEDDINGS_PROVIDER", "huggingface").lower()
        cls.LLM_PROVIDER = _get_secret("LLM_PROVIDER", "openai").lower()

        # Modelos
        cls.HF_EMBEDDING_MODEL = _get_secret("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        cls.OPENAI_MODEL_NAME = _get_secret("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        cls.LOCAL_LLM_MODEL = _get_secret("LOCAL_LLM_MODEL", "llama3")

        # API Keys
        cls.OPENAI_API_KEY = _get_secret("OPENAI_API_KEY", "")

    @classmethod
    def validate(cls):
        """Valida configuraciones necesarias y loguea advertencias (no crashea al importar)."""
        cls._load()  # Recargar valores en el momento de la validación

        if not os.path.exists(cls.PDF_DIRECTORY):
            logger.warning(f"El directorio de PDFs no existe: {cls.PDF_DIRECTORY}")

        if cls.EMBEDDINGS_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY es requerida cuando EMBEDDINGS_PROVIDER es 'openai'")

        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY es requerida cuando LLM_PROVIDER es 'openai'")

        logger.info("Configuración cargada.")
        return cls


# Carga inicial de valores (sin crashear si falta la API key)
Config._load()
