# Timonel RAG 📘

[![CI](https://github.com/TURRIvalentin/Timonel-AI-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/TURRIvalentin/Timonel-AI-Agent/actions/workflows/ci.yml)
[![Docker](https://github.com/TURRIvalentin/Timonel-AI-Agent/actions/workflows/docker.yml/badge.svg)](https://github.com/TURRIvalentin/Timonel-AI-Agent/actions/workflows/docker.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Sistema de Recuperación Aumentada por Generación (RAG) para consultar de manera inteligente un conjunto de documentos PDF locales. Utiliza LangChain, ChromaDB y Streamlit para brindar una interfaz interactiva y fácil de usar.

---

## 🚀 API REST

```bash
# Local
uvicorn src.api.main:app --reload
# Swagger UI → http://localhost:8000/docs

# Docker
cp .env.example .env   # agregar OPENAI_API_KEY
docker compose up      # API disponible en http://localhost:8000
```

Endpoints: `GET /health` · `POST /ingest` · `POST /query`

## Características

*   **Ingesta Robusta**: Escanea y carga PDFs automáticamente desde una carpeta definida, tolerando archivos corruptos para no detener el proceso.
*   **Vector Store Local**: Utiliza ChromaDB para guardar y persistir los embeddings localmente.
*   **Flexible**: Soporta embeddings mediante HuggingFace (Open-Source y local) u OpenAI, y LLMs locales (Ollama) o mediante API (OpenAI).
*   **Fuentes Documentales**: Cada respuesta incluye la cita del documento y página de donde se extrajo la información.
*   **Interfaz Dual**: Consultas rápidas desde la terminal (CLI) o a través de una interfaz web con Streamlit.

---

## 🛠️ Requisitos Previos

1.  Python 3.9 o superior.
2.  Tus documentos PDF ubicados en un directorio (por defecto configurado para `C:\Users\valen\OneDrive\Escritorio\Timonel`).

---

## 🚀 Instalación y Configuración

### 1. Clonar o descargar el proyecto e ir al directorio
```bash
cd Timonel-RAG
```

### 2. Crear y activar un entorno virtual (Recomendado)
**En Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
Copia el archivo de ejemplo para crear tu propio `.env`:
```bash
cp .env.example .env
```
Edita el archivo `.env` para ajustar las rutas de tus PDFs y si usarás OpenAI o un proveedor local.
*   Si usas **OpenAI**, asegúrate de colocar tu `OPENAI_API_KEY`.
*   Si usas **Local (HuggingFace + Ollama)**, asegúrate de tener Ollama instalado y el modelo (ej. llama3) descargado (`ollama run llama3`).

---

## ⚙️ Uso Paso a Paso

### Paso 1: Ingesta de Documentos
Antes de poder hacer preguntas, necesitas procesar los PDFs y crear la base de datos vectorial. Ejecuta el script de ingesta:

```bash
python src/ingest.py
```
*Este proceso leerá todos los PDFs, extraerá el texto, lo dividirá en trozos (chunks), calculará los embeddings y los guardará en la carpeta `./chroma_db`.*

### Paso 2: Consultar al Sistema

Tienes dos formas de consultar tus documentos:

#### Opción A: Interfaz de Línea de Comandos (CLI)
Útil para pruebas rápidas en la terminal.

```bash
python src/query.py "¿Cuál es el tema principal del documento X?"
```

#### Opción B: Interfaz Web Interactiva (Streamlit)
La mejor opción para uso continuo. Lanza la aplicación web con:

```bash
streamlit run app.py
```
*Esto abrirá automáticamente una pestaña en tu navegador web donde podrás interactuar con el sistema RAG, hacer preguntas y visualizar las fuentes.*

---

## 📁 Estructura del Proyecto

```text
Timonel-RAG/
├── .env                  # (No incluido en repo) Tus variables de configuración
├── .env.example          # Plantilla de variables de entorno
├── requirements.txt      # Dependencias de Python
├── README.md             # Esta documentación
├── app.py                # Interfaz web con Streamlit
└── src/
    ├── __init__.py
    ├── config.py         # Carga de variables del .env y validaciones
    ├── ingest.py         # Pipeline de lectura y partición de PDFs
    └── query.py          # Lógica de retrieval y QA Chain
```

## 📝 Notas Adicionales
*   **Calidad de respuestas:** La calidad dependerá del LLM seleccionado. `gpt-3.5-turbo` o superior suele dar resultados excelentes. Modelos locales como `llama3` funcionan muy bien pero requieren recursos (RAM/GPU) en la máquina local.
*   **Actualización de documentos:** Si agregas nuevos PDFs a la carpeta, vuelve a ejecutar `python src/ingest.py` para actualizar la base de datos vectorial.
