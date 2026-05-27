# src/semantica.py
import os
import chromadb
from sentence_transformers import SentenceTransformer

class MotorSemantico:
    def __init__(self, ruta_db="db/vector_store", usar_multiproceso=True):
        """
        Inicializa el cliente de ChromaDB persistente en disco
        y carga el modelo de embeddings.
        """
        # 1. Configurar la base de datos vectorial persistente
        os.makedirs(ruta_db, exist_ok=True)
        self.cliente = chromadb.PersistentClient(path=ruta_db)

        self.dispositivo = self._detectar_dispositivo()
        self.usar_multiproceso = bool(usar_multiproceso)
        
        # 2. Cargar el modelo de HuggingFace (se descarga automáticamente la primera vez)
        print(f"🤖 Cargando modelo de embeddings (all-MiniLM-L6-v2) en {self.dispositivo}...")
        self.modelo = SentenceTransformer("all-MiniLM-L6-v2", device=self.dispositivo)
        
        # 3. Crear o recuperar la colección dentro de ChromaDB
        # Usamos una función de distancia de coseno para el ranking
        self.coleccion = self.cliente.get_or_create_collection(
            name="reuters_documents", 
            metadata={"hnsw:space": "cosine"}
        )

    def _detectar_dispositivo(self):
        """Selecciona GPU si está disponible; si no, CPU."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _encode_lote(self, textos, batch_size):
        """Codifica textos priorizando GPU y multiproceso en CPU para lotes grandes."""
        if self.dispositivo == "cpu" and self.usar_multiproceso and len(textos) >= 2000:
            pool = self.modelo.start_multi_process_pool()
            try:
                embeddings = self.modelo.encode_multi_process(
                    textos,
                    pool,
                    batch_size=batch_size,
                )
            finally:
                self.modelo.stop_multi_process_pool(pool)
            return embeddings

        return self.modelo.encode(
            textos,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
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
        batch_size = 512 if self.dispositivo == "cuda" else 128
        total_docs = len(textos)

        # En CPU con corpus grande conviene codificar en multiproceso una sola vez.
        if self.dispositivo == "cpu" and self.usar_multiproceso and total_docs >= 2000:
            embeddings = self._encode_lote(textos, batch_size=batch_size).tolist()
            self.coleccion.add(ids=ids, embeddings=embeddings, documents=textos)
            print(f"✅ Base de datos vectorial construida y guardada exitosamente ({total_docs} documentos).")
            return
        
        for i in range(0, total_docs, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_textos = textos[i:i + batch_size]
            
            # 1. Generar los vectores matemáticos de este lote
            embeddings = self._encode_lote(batch_textos, batch_size=batch_size).tolist()
            
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
        query_embedding = self.modelo.encode(
            consulta,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
        
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