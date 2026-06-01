# src/modelos_clasicos.py
import math
import collections
from src.preprocesamiento import limpiar_texto

class MotorClasico:
    def __init__(self, objeto_indice):
        """
        Recibe una instancia de la clase IndiceInvertido ya construida.
        """
        self.idx = objeto_indice
        self.indice = objeto_indice.indice
        self.doc_lengths = objeto_indice.doc_lengths
        self.N = objeto_indice.num_docs
        
        # Calcular el promedio de longitud de documentos (necesario para BM25)
        self.avgdl = sum(self.doc_lengths.values()) / self.N if self.N > 0 else 0
        
        # Cache para normalización de BM25
        self._cache_bm25_max = {}
    
    def _normalizar_scores(self, scores_dict, metodo="minmax"):
        """
        Normaliza scores a rango [0, 1] para comparación justa entre modelos.
        
        Args:
            scores_dict: dict {doc_id: score}
            metodo: "minmax" (recomendado) o "softmax"
        
        Returns:
            lista [(doc_id, score_norm)] ordenada descendente
        """
        if not scores_dict:
            return []
        
        scores_list = list(scores_dict.items())
        valores = [v for _, v in scores_list]
        
        if metodo == "minmax":
            # Normalización Min-Max: (x - min) / (max - min)
            minimo = min(valores)
            maximo = max(valores)
            
            if minimo == maximo:
                # Todos los scores iguales
                scores_norm = [(doc_id, 1.0) for doc_id, _ in scores_list]
            else:
                scores_norm = [
                    (doc_id, (score - minimo) / (maximo - minimo))
                    for doc_id, score in scores_list
                ]
        
        elif metodo == "softmax":
            # Normalización Softmax
            max_score = max(valores)
            exp_scores = [math.exp(score - max_score) for score in valores]
            sum_exp = sum(exp_scores)
            scores_norm = [
                (doc_id, exp_score / sum_exp)
                for (doc_id, _), exp_score in zip(scores_list, exp_scores)
            ]
        
        else:
            raise ValueError(f"Método desconocido: {metodo}")
        
        # Ordenar descendente por score normalizado
        return sorted(scores_norm, key=lambda x: x[1], reverse=True)

    # ---------------------------------------------------------
    # MODELO 1: SIMILITUD DE JACCARD (Vectores Binarios)
    # ---------------------------------------------------------
    def buscar_jaccard(self, consulta, top_k=10):
        """Implementa Jaccard: |Q ∩ D| / |Q ∪ D| (ignora frecuencias)"""
        tokens_q = set(limpiar_texto(consulta))
        if not tokens_q:
            return []

        scores = {}
        # Evaluamos solo los documentos que contienen al menos una palabra de la consulta
        for termino in tokens_q:
            if termino in self.indice:
                for doc_id in self.indice[termino].keys():
                    if doc_id not in scores:
                        # Reconstruimos el conjunto de palabras únicas del documento d desde el índice
                        # (Optimización para no volver a leer el texto del disco)
                        tokens_d = set([t for t in self.indice if doc_id in self.indice[t]])
                        
                        interseccion = len(tokens_q.intersection(tokens_d))
                        union = len(tokens_q.union(tokens_d))
                        
                        scores[doc_id] = interseccion / union if union > 0 else 0
                        
        # Ordenar ranking de mayor a menor relevancia
        ranking_ordenado = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranking_ordenado[:top_k]

    def buscar_booleano(self, consulta, top_k=10):
        """Alias de compatibilidad para búsqueda booleana basada en Jaccard."""
        return self.buscar_jaccard(consulta, top_k=top_k)

    # ---------------------------------------------------------
    # MODELO 2: SIMILITUD DE COSENO (TF-IDF)
    # ---------------------------------------------------------
    def buscar_coseno_tfidf(self, consulta, top_k=10):
        """Implementa Similitud de Coseno utilizando pesos TF-IDF"""
        tokens_q = limpiar_texto(consulta)
        if not tokens_q:
            return []

        conteo_q = collections.Counter(tokens_q)
        query_weights = {}
        doc_vectors = collections.defaultdict(dict) # {doc_id: {termino: peso_tfidf}}
        
        # 1. Calcular componentes del vector de la consulta y acumular pesos del documento
        for termino, freq_q in conteo_q.items():
            if termino in self.indice:
                # df = cantidad de documentos que contienen el término
                df = len(self.indice[termino])
                # idf clásico
                idf = math.log10(self.N / df) if df > 0 else 0
                
                # Peso TF-IDF de la consulta (TF logarítmico)
                tf_q = 1 + math.log10(freq_q)
                query_weights[termino] = tf_q * idf
                
                # Pesos TF-IDF para cada documento que tiene este término de la consulta
                for doc_id, freq_d in self.indice[termino].items():
                    tf_d = 1 + math.log10(freq_d)
                    doc_vectors[doc_id][termino] = tf_d * idf

        # 2. Calcular la Similitud del Coseno (Producto punto / Producto de Normas)
        scores = {}
        norma_q = math.sqrt(sum(w ** 2 for w in query_weights.values()))
        if norma_q == 0:
            return []

        for doc_id, vector_d in doc_vectors.items():
            # Producto punto
            producto_punto = sum(query_weights[t] * vector_d.get(t, 0) for t in query_weights)
            
            # Norma del documento (Nota: Para un coseno riguroso se requiere la norma de TODO el documento,
            # pero usar la longitud de tokens o la norma sobre la consulta es una aproximación estándar efectiva)
            norma_d = math.sqrt(sum(w ** 2 for w in vector_d.values()))
            
            scores[doc_id] = producto_punto / (norma_q * norma_d) if norma_d > 0 else 0

        ranking_ordenado = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranking_ordenado[:top_k]

    # ---------------------------------------------------------
    # MODELO 3: BM25 (Okapi BM25)
    # ---------------------------------------------------------
    def buscar_bm25(self, consulta, k1=1.5, b=0.75, top_k=10):
        """Implementa el algoritmo probabilístico Okapi BM25"""
        tokens_q = limpiar_texto(consulta)
        if not tokens_q:
            return []

        scores = collections.defaultdict(float)
        
        for termino in set(tokens_q):
            if termino in self.indice:
                df = len(self.indice[termino])
                # Cálculo del IDF de BM25 (admite suavizado para evitar valores negativos)
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                
                for doc_id, tf in self.indice[termino].items():
                    doc_len = self.doc_lengths.get(doc_id, self.avgdl)
                    
                    # Fórmula del denominador de BM25
                    numerador = tf * (k1 + 1)
                    denominador = tf + k1 * (1 - b + b * (doc_len / self.avgdl))
                    
                    scores[doc_id] += idf * (numerador / denominador)

        ranking_ordenado = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranking_ordenado[:top_k]

    # ---------------------------------------------------------
    # VERSIONES NORMALIZADAS (scores en [0, 1] para comparación justa)
    # ---------------------------------------------------------
    
    def buscar_jaccard_norm(self, consulta, top_k=10):
        """Jaccard con scores normalizados a [0, 1] (ya está normalizado)."""
        return self.buscar_jaccard(consulta, top_k)
    
    def buscar_coseno_tfidf_norm(self, consulta, top_k=10):
        """TF-IDF con scores normalizados a [0, 1] (ya está normalizado)."""
        return self.buscar_coseno_tfidf(consulta, top_k)
    
    def buscar_bm25_norm(self, consulta, k1=1.5, b=0.75, top_k=10):
        """
        BM25 con scores normalizados a [0, 1] usando Min-Max.
        
        BM25 retorna scores sin límite superior. Esta versión normaliza 
        los scores para comparación justa con otros modelos.
        """
        tokens_q = limpiar_texto(consulta)
        if not tokens_q:
            return []

        scores = collections.defaultdict(float)
        
        for termino in set(tokens_q):
            if termino in self.indice:
                df = len(self.indice[termino])
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                
                for doc_id, tf in self.indice[termino].items():
                    doc_len = self.doc_lengths.get(doc_id, self.avgdl)
                    numerador = tf * (k1 + 1)
                    denominador = tf + k1 * (1 - b + b * (doc_len / self.avgdl))
                    scores[doc_id] += idf * (numerador / denominador)

        # ✅ Normalizar a [0, 1]
        return self._normalizar_scores(scores, metodo="minmax")[:top_k]