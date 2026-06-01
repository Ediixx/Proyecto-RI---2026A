"""
Módulo de evaluación para Sistema de Recuperación de Información.
Calcula métricas: Precision, Recall, F1, AP y MAP
"""

from collections import defaultdict


def calcular_precision_recall(doc_recuperados, doc_relevantes):
    """
    Calcula Precision y Recall para una consulta.
    
    Args:
        doc_recuperados: lista/set de IDs de documentos recuperados
        doc_relevantes: lista/set de IDs de documentos relevantes
    
    Returns:
        (precision, recall): tupla con métricas
    """
    doc_recuperados = set(str(d) for d in doc_recuperados)
    doc_relevantes = set(str(d) for d in doc_relevantes)
    
    interseccion = len(doc_recuperados & doc_relevantes)
    
    precision = interseccion / len(doc_recuperados) if doc_recuperados else 0.0
    recall = interseccion / len(doc_relevantes) if doc_relevantes else 0.0
    
    return precision, recall


def calcular_f1(precision, recall):
    """Calcula F1-score a partir de precision y recall."""
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calcular_average_precision(ranking, doc_relevantes, top_k=None):
    """
    Calcula Average Precision (AP) para una consulta.
    
    AP = (1/|R|) * Σ(P(k) * rel(k))
    donde:
    - |R| es el número de documentos relevantes
    - P(k) es la precisión en posición k
    - rel(k) es 1 si el documento en posición k es relevante, 0 si no
    
    Args:
        ranking: lista de tuplas [(doc_id, score), ...]
        doc_relevantes: conjunto de IDs de documentos relevantes
        top_k: si se especifica, solo considera los primeros top_k documentos
    
    Returns:
        ap: valor de Average Precision (0-1)
    """
    if not ranking or not doc_relevantes:
        return 0.0
    
    doc_relevantes = set(str(d) for d in doc_relevantes)
    
    if top_k:
        ranking = ranking[:top_k]
    
    ap = 0.0
    num_relevantes_encontrados = 0
    
    for posicion, (doc_id, _) in enumerate(ranking, 1):
        if str(doc_id) in doc_relevantes:
            num_relevantes_encontrados += 1
            precision_en_k = num_relevantes_encontrados / posicion
            ap += precision_en_k
    
    if not doc_relevantes:
        return 0.0
    
    # Normalizar por el total de documentos relevantes
    ap = ap / len(doc_relevantes)
    
    return ap


def evaluar_modelo(buscar_func, qrels, top_k=5, progress_callback=None):
    """
    Evalúa un modelo de búsqueda usando un conjunto de consultas QREL.
    
    Args:
        buscar_func: función que realiza la búsqueda (consulta, top_k) -> [(doc_id, score), ...]
        qrels: diccionario {consulta: [doc_ids_relevantes, ...]}
        top_k: número de documentos a recuperar por consulta
        progress_callback: función opcional para reportar progreso (llamada con (actual, total, mensaje))
    
    Returns:
        diccionario con métricas por consulta y agregadas:
        {
            'por_consulta': [
                {
                    'query': consulta,
                    'precision': float,
                    'recall': float,
                    'f1': float,
                    'ap': float,
                    'num_relevantes': int,
                    'num_recuperados': int
                },
                ...
            ],
            'agregadas': {
                'precision_media': float,
                'recall_media': float,
                'f1_media': float,
                'map': float,  # Mean Average Precision
                'num_consultas': int
            }
        }
    """
    metricas_por_consulta = []
    consultas_lista = list(qrels.items())
    total_consultas = len(consultas_lista)
    
    for idx, (consulta, doc_relevantes) in enumerate(consultas_lista, 1):
        doc_relevantes = set(str(d) for d in doc_relevantes)
        
        if not doc_relevantes:
            continue
        
        # Reportar progreso
        if progress_callback:
            msg = f"Evaluando: {str(consulta)[:40]}... ({idx}/{total_consultas})"
            progress_callback(idx, total_consultas, msg)
        
        # Ejecutar búsqueda
        try:
            ranking = buscar_func(str(consulta), top_k=top_k)
        except Exception as e:
            print(f"⚠️  Error al buscar '{consulta}': {e}")
            continue
        
        doc_recuperados = [str(doc_id) for doc_id, _ in ranking]
        
        # Calcular métricas
        precision, recall = calcular_precision_recall(doc_recuperados, doc_relevantes)
        f1 = calcular_f1(precision, recall)
        ap = calcular_average_precision(ranking, doc_relevantes, top_k=top_k)
        
        metricas_por_consulta.append({
            'query': str(consulta)[:50],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,
            'num_relevantes': len(doc_relevantes),
            'num_recuperados': len(doc_recuperados)
        })
    
    # Calcular agregados
    if not metricas_por_consulta:
        return {
            'por_consulta': [],
            'agregadas': {
                'precision_media': 0.0,
                'recall_media': 0.0,
                'f1_media': 0.0,
                'map': 0.0,
                'num_consultas': 0
            }
        }
    
    n = len(metricas_por_consulta)
    agregadas = {
        'precision_media': sum(m['precision'] for m in metricas_por_consulta) / n,
        'recall_media': sum(m['recall'] for m in metricas_por_consulta) / n,
        'f1_media': sum(m['f1'] for m in metricas_por_consulta) / n,
        'map': sum(m['ap'] for m in metricas_por_consulta) / n,
        'num_consultas': n
    }
    
    return {
        'por_consulta': metricas_por_consulta,
        'agregadas': agregadas
    }


def comparar_modelos(modelos, qrels, top_k=5, progress_callback=None):
    """
    Compara múltiples modelos de búsqueda.
    
    Args:
        modelos: diccionario {nombre_modelo: buscar_func}
        qrels: diccionario de consultas relevantes
        top_k: número de documentos a recuperar
        progress_callback: función opcional para reportar progreso
    
    Returns:
        diccionario con resultados de evaluación de cada modelo
    """
    resultados = {}
    modelos_lista = list(modelos.items())
    total_modelos = len(modelos_lista)
    
    for idx, (nombre_modelo, buscar_func) in enumerate(modelos_lista, 1):
        try:
            if progress_callback:
                def cb_modelo(actual, total, msg):
                    progreso_global = (idx - 1 + actual/total) / total_modelos
                    progress_callback(progreso_global, 1.0, f"[{idx}/{total_modelos}] {nombre_modelo}: {msg}")
            else:
                cb_modelo = None
            
            resultados[nombre_modelo] = evaluar_modelo(buscar_func, qrels, top_k, progress_callback=cb_modelo)
        except Exception as e:
            print(f"❌ Error evaluando modelo '{nombre_modelo}': {e}")
            resultados[nombre_modelo] = None
    
    return resultados


def generar_reporte_comparativo(comparacion, metricas=['precision_media', 'recall_media', 'map']):
    """
    Genera un reporte comparativo de modelos.
    
    Args:
        comparacion: diccionario de comparación (salida de comparar_modelos)
        metricas: lista de métricas a mostrar
    
    Returns:
        diccionario con ranking de modelos por métrica
    """
    reporte = {}
    
    for metrica in metricas:
        ranking = []
        for modelo, resultado in comparacion.items():
            if resultado is None:
                continue
            valor = resultado['agregadas'].get(metrica, 0)
            ranking.append((modelo, valor))
        
        ranking.sort(key=lambda x: x[1], reverse=True)
        reporte[metrica] = ranking
    
    return reporte
