import os
import chromadb
from pathlib import Path
from pypdf import PdfReader
from config import get_settings

def extraer_texto_pdf(ruta_pdf):
    """Extrae todo el texto de un archivo PDF."""
    reader = PdfReader(ruta_pdf)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() + "\n"
    return texto

def preparar_bcl():
    settings = get_settings()
    
    # 1. Asegurar que el directorio de persistencia existe
    persist_path = Path(settings.chroma_persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Inicializar cliente de ChromaDB
    client = chromadb.PersistentClient(path=str(persist_path))
    
    # 3. Crear o obtener la colección
    collection = client.get_or_create_collection(name=settings.bcl_collection_name)
    
    # 4. Buscar archivos para ingestar (.txt y .pdf)
    data_folder = Path("data_ingesta")
    if not data_folder.exists():
        data_folder.mkdir()
        print("📁 Carpeta 'data_ingesta' creada. Coloca tus archivos .txt o .pdf allí.")
        return

    archivos = list(data_folder.glob("*.txt")) + list(data_folder.glob("*.pdf"))
    if not archivos:
        print("⚠️ No hay archivos para procesar en 'data_ingesta'.")
        return

    print(f"🚀 Procesando {len(archivos)} archivos...")

    for archivo in archivos:
        try:
            if archivo.suffix.lower() == ".pdf":
                contenido = extraer_texto_pdf(archivo)
            else:
                with open(archivo, "r", encoding="utf-8") as f:
                    contenido = f.read()

            if not contenido.strip():
                print(f"⏩ Saltado (vacío): {archivo.name}")
                continue

            # Añadir a la base de datos
            collection.add(
                documents=[contenido],
                ids=[archivo.name],
                metadatas=[{"source": archivo.name, "type": archivo.suffix[1:]}]
            )
            print(f"✅ Ingestado: {archivo.name}")
        except Exception as e:
            print(f"❌ Error procesando {archivo.name}: {e}")

    print("\n✨ Base de Datos BCL lista y actualizada con archivos locales.")

if __name__ == "__main__":
    preparar_bcl()
