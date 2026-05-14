# NOVA Backend v2.0

## Descripción General
NOVA Backend es el motor de orquestación multi-agente diseñado para asistir en flujos de trabajo periodísticos de alta complejidad. Utiliza el framework CrewAI para coordinar agentes especializados que garantizan la calidad editorial, ética y dialéctica de los contenidos generados, integrando capacidades de Generación Aumentada por Recuperación (RAG) mediante una base de conocimientos técnica (BCL).

El sistema implementa una arquitectura de "Caja Blanca" (White Box) que permite la trazabilidad en tiempo real de los procesos de pensamiento y ejecución de cada agente a través de eventos de WebSocket.

## Características Principales
*   **Orquestación Multi-Agente:** Flujo secuencial compuesto por agentes de Redacción Editorial, Validación Ética, Análisis Dialéctico y Adaptación Multimodal.
*   **Active RAG (BCL Integration):** Recuperación dinámica de información desde documentos normativos y manuales de estilo.
*   **Transparencia en Tiempo Real:** Streaming de eventos internos de los agentes hacia el frontend mediante Socket.IO.
*   **Persistencia de Sesiones:** Almacenamiento local de interacciones y estados de sesión mediante SQLite.
*   **Arquitectura Escalable:** Basado en FastAPI y Pydantic para garantizar un alto rendimiento y validación robusta de datos.

## Stack Tecnológico
*   **Lenguaje:** Python 3.10+
*   **Framework de Agentes:** CrewAI
*   **API Framework:** FastAPI
*   **Comunicación en Tiempo Real:** Socket.IO (ASGI)
*   **Base de Datos Vectorial:** ChromaDB
*   **Base de Datos Relacional:** SQLite (SQLAlchemy)
*   **Proveedores de LLM compatibles:** OpenAI, Anthropic, Gemini y proveedores compatibles con OpenAI.

## Estructura del Proyecto
```text
NOVA_Backend/
├── agents/             # Definiciones de roles y comportamiento de agentes.
├── bcl/                # Lógica de carga y consulta de base de conocimientos.
├── data_ingesta/       # Documentos fuente para el RAG.
├── llm/                # Configuración de proveedores de modelos de lenguaje.
├── orchestrator/       # Lógica de CrewAI (tareas y orquestación).
├── services/           # Bus de mensajes y persistencia de datos.
├── tools/              # Herramientas personalizadas para los agentes.
├── main.py             # Punto de entrada de la aplicación FastAPI.
└── config.py           # Gestión de variables de entorno y configuración.
```

## Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd NOVA_Backend
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Variables de Entorno
Cree un archivo `.env` en la raíz del proyecto basado en el siguiente esquema:
```env
APP_NAME="Nova Backend"
ENVIRONMENT="development"
CORS_ALLOWED_ORIGINS="*"

LLM_PROVIDER="openai"
LLM_MODEL="gpt-4-turbo"
OPENAI_API_KEY="tu_clave_aqui"

DATABASE_URL="sqlite:///./nova.db"
CHROMA_PERSIST_PATH="./chroma_db"
```

## Ejecución

### Desarrollo
Para iniciar el servidor con recarga automática:
```bash
python main.py
```
El servidor estará disponible en `http://localhost:8000`. Puede acceder a la documentación interactiva de la API en `http://localhost:8000/docs`.

### Ingesta de Datos (RAG)
Para actualizar la base de conocimientos vectorial:
```bash
python ingest_bcl.py
```

## Protocolo de Comunicación (Socket.IO)
El backend emite eventos bajo el nombre `agent_event`. Los clientes deben suscribirse a este evento para recibir las actualizaciones del estado de los agentes y los fragmentos de pensamiento (logs de ejecución).

## Licencia
Este proyecto es propiedad privada de NOVA. Todos los derechos reservados.