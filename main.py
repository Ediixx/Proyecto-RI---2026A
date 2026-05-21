# main.py
import os
from src.indexacion import IndiceInvertido
from src.modelos_clasicos import MotorClasico

ruta_indice = "db/indice_invertido.pkl"

# 1. Cargar el índice que ya guardamos con éxito
if os.path.exists(ruta_indice):
    indice_cargado = IndiceInvertido.cargar_de_disco(ruta_indice)
    motor = MotorClasico(indice_cargado)
    
    # 2. Definir una consulta típicamente financiera de Reuters
    consulta = "tariffs on Japanese electronics goods"
    print(f"🔍 Buscando con TF-IDF + Coseno para: '{consulta}'...\n")
    
    # 3. Recuperar el Top 5
    resultados_tfidf = motor.buscar_coseno_tfidf(consulta, top_k=5)
    
    # Imprimir el ranking estructurado [(doc_id, score_coseno)]
    for puesto, (doc_id, score) in enumerate(resultados_tfidf, 1):
        print(f"{puesto}. Documento ID: {doc_id} -> Score de Coseno: {score:.4f}")
else:
    print("⚠️ Primero debes construir el índice ejecutando el script de inicialización.")