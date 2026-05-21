# src/indexacion.py
import collections
import pickle
import os
from src.preprocesamiento import limpiar_texto

class IndiceInvertido:
    def __init__(self):
        # Estructura: { "termino": { "doc_id_1": frecuencia, "doc_id_2": frecuencia } }
        self.indice = collections.defaultdict(dict)
        # Guardamos la longitud de cada documento (número de tokens) para BM25
        self.doc_lengths = {}
        # Guardamos el total de documentos indexados
        self.num_docs = 0

    def construir_indice(self, corpus):
        """
        Recibe el diccionario del corpus {doc_id: texto_completo}
        y llena el índice invertido.
        """
        print("🛠️  Construyendo el índice invertido...")
        self.num_docs = len(corpus)
        
        for doc_id, texto in corpus.items():
            # Pasamos el texto por nuestro módulo de preprocesamiento
            tokens = limpiar_texto(texto)
            
            # Registrar la longitud del documento
            self.doc_lengths[doc_id] = len(tokens)
            
            # Contar frecuencias de términos en este documento específico
            conteo_terminos = collections.Counter(tokens)
            
            for termino, freq in conteo_terminos.items():
                self.indice[termino][doc_id] = freq
                
        print(f"✅ Índice construido con {len(self.indice)} términos únicos.")

    def guardar_en_disco(self, ruta_archivo):
        """Guarda el objeto indexado en un archivo binario."""
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        with open(ruta_archivo, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Índice guardado exitosamente en: {ruta_archivo}")

    @staticmethod
    def cargar_de_disco(ruta_archivo):
        """Carga el objeto indexado desde el disco."""
        with open(ruta_archivo, 'rb') as f:
            return pickle.load(f) # Nota: Aquí usamos pickle.load(f) al ejecutarlo