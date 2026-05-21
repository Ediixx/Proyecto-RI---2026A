# src/semantica.py
import os
import chromadb
from sentence_transformers import SentenceTransformer

class MotorSemantico:
    def __init__(self, ruta_db="db/vector_store"):
        """
        Inicializa el cliente de ChromaDB persistente en disco
        y carga el modelo de embeddings.
        """
        # 1. Configurar la base de datos vectorial persistente
        os.makedirs(ruta_db, exist_ok=True)
        self.cliente = chromadb.PersistentClient(path=ruta_db)
        
        # 2. Cargar el modelo de HuggingFace (se descarga automáticamente la primera vez)
        print("🤖 Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
        self.modelo = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 3. Crear o recuperar la colección dentro de ChromaDB
        # Usamos una función de distancia de coseno para el ranking
        self.coleccion = self.cliente.get_or_create_collection(
            name="reuters_documents", 
            metadata={"hnsw:space": "cosine"}
        )

    def indexar_corpus_vectorial(self, corpus):
        """
        Recibe el corpus {doc_id: texto_completo}, genera los embeddings
        en lotes (batches) y los almacena en ChromaDB.
        """
        # Si la colección ya tiene documentos, evitamos reindexar
        if self.coleccion.count() > 0:
            print(f"📖 Base vectorial detectada con {self.coleccion.count()} documentos. Saltando indexación...")
            return

        print("⚡ Generando embeddings e indexando en la base vectorial (Esto puede tomar unos minutos)...")
        
        ids = list(corpus.keys())
        textos = list(corpus.values())
        
        # Procesamos en lotes para no saturar la memoria RAM
        batch_size = 256
        total_docs = len(textos)
        
        for i in range(0, total_docs, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_textos = textos[i:i + batch_size]
            
            # 1. Generar los vectores matemáticos de este lote
            embeddings = self.modelo.encode(batch_textos, show_progress_bar=False).tolist()
            
            # 2. Guardar en ChromaDB
            self.coleccion.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_textos # Guardamos el texto para poder mostrarlo en la interfaz
            )
            print(f"🔄 Indexados {min(i + batch_size, total_docs)} / {total_docs} documentos...")
            
        print("✅ Base de datos vectorial construida y guardada exitosamente.")

    def buscar_semantica(self, consulta, top_k=10):
        """
        Convierte la consulta en vector y busca los documentos más similares en ChromaDB.
        """
        if not consulta.strip():
            return []

        # 1. Convertir la consulta del usuario a embedding
        query_embedding = self.modelo.encode(consulta).tolist()
        
        # 2. Consultar la base de datos
        resultados = self.coleccion.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # 3. Formatear la salida para que sea idéntica a los modelos clásicos: [(doc_id, score), ...]
        ranking_semantico = []
        if resultados["ids"] and resultados["distances"]:
            # Nota: ChromaDB con 'cosine' a veces regresa la distancia. 
            # Score de similitud = 1 - distancia
            for doc_id, distancia in zip(resultados["ids"][0], resultados["distances"][0]):
                score_similitud = 1 - distancia
                ranking_semantico.append((doc_id, score_similitud))
                
        return ranking_semantico