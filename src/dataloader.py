# src/dataloader.py
import pandas as pd
import os

def cargar_corpus_completo(carpeta_data):
    """
    Carga y combina los archivos de entrenamiento y prueba de ModApte.
    Retorna:
        - corpus: { "doc_id": "titulo + cuerpo" }
        - metadata_topics: { "doc_id": ["topic1", "topic2"] }
    """
    # Definimos los archivos que sí vamos a indexar
    archivos_a_cargar = [
        os.path.join(carpeta_data, "ModApte_train.csv"),
        os.path.join(carpeta_data, "ModApte_test.csv")
    ]
    
    corpus = {}
    metadata_topics = {}
    
    for archivo in archivos_a_cargar:
        if not os.path.exists(archivo):
            print(f"⚠️ Advertencia: No se encontró el archivo {archivo}")
            continue
            
        print(f"📦 Cargando documentos desde: {os.path.basename(archivo)}...")
        
        # Leer el CSV (Reuters usa comas, pero a veces el texto interno tiene saltos)
        df = pd.read_csv(archivo, usecols=['new_id', 'title', 'text', 'topics'])
        df = df.dropna(subset=['new_id', 'text'])
        
        # src/dataloader.py

        for _, row in df.iterrows():
            # .strip('"') elimina las comillas dobles que causan el error de conversión
            id_limpio = str(row['new_id']).strip('"')
            
            # Ahora la conversión a int y luego a string funcionará sin problemas
            doc_id = str(int(id_limpio))
            
            titulo = str(row['title']).strip() if pd.notna(row['title']) else ""
            cuerpo = str(row['text']).strip()
            
            # Combinamos título y cuerpo para mejorar la indexación
            texto_completo = f"{titulo}\n{cuerpo}" if titulo else cuerpo
            corpus[doc_id] = texto_completo
            
            # Procesar los tópicos (vienen separados por comas o espacios en el CSV)
            # Ej: "earn,acq" -> ["earn", "acq"]
            topicos_raw = row['topics']
            if pd.notna(topicos_raw) and str(topicos_raw).strip():
                # Limpieza básica para convertirlos en una lista de categorías
                lista_topicos = [t.strip() for t in str(topicos_raw).split(',') if t.strip()]
                metadata_topics[doc_id] = lista_topicos
            else:
                metadata_topics[doc_id] = []
                
    print(f"✅ Carga finalizada. Total de documentos indexables: {len(corpus)}")
    return corpus, metadata_topics