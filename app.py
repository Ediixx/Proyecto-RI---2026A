# app.py
import streamlit as tf_web  # Alias para evitar conflictos de nombres
import os
from src.indexacion import IndiceInvertido
from src.modelos_clasicos import MotorClasico
from src.dataloader import cargar_corpus_completo

# Configuración de la página web (Pestaña del navegador)
tf_web.set_page_config(
    page_title="Motor de Búsqueda Reuters-21578",
    page_icon="🔍",
    layout="wide"
)

# Ruta del índice persistido
RUTA_INDICE = "db/indice_invertido.pkl"
RUTA_DATA = "data"

# Estilizar la interfaz usando el estado de sesión de Streamlit para cargar el motor una sola vez
@tf_web.cache_resource
def inicializar_motor():
    """Carga el índice existente o lo construye si no se encuentra."""
    if os.path.exists(RUTA_INDICE) and os.path.getsize(RUTA_INDICE) > 0:
        indice = IndiceInvertido.cargar_de_disco(RUTA_INDICE)
    else:
        # Si por alguna razón no está, lo construye al vuelo
        corpus, _ = cargar_corpus_completo(RUTA_DATA)
        indice = IndiceInvertido()
        indice.construir_indice(corpus)
        indice.guardar_en_disco(RUTA_INDICE)
    
    # También necesitamos el corpus original para mostrar el texto de los resultados
    # Lo volvemos a cargar de forma eficiente
    corpus, _ = cargar_corpus_completo(RUTA_DATA)
    return MotorClasico(indice), corpus

# Inicializar los datos del backend
try:
    motor, corpus_textos = inicializar_motor()
except Exception as e:
    tf_web.error(f"Error al cargar el corpus o el índice: {e}")
    tf_web.stop()

# --- DISEÑO DE LA INTERFAZ GRÁFICA (FRONTEND) ---

tf_web.title("🔍 Sistema de Recuperación de Información Financiera")
tf_web.subheader("Análisis comparativo de modelos clásicos sobre el Corpus Reuters-21578")
tf_web.markdown("---")

# Crear una disposición de dos columnas para los controles principales
col1, col2 = tf_web.columns([3, 1])

with col1:
    # Barra de entrada de texto para que el usuario escriba su consulta libre
    query_usuario = tf_web.text_input(
        "Introduce los términos de tu búsqueda financiera:",
        placeholder="Ejemplo: tariffs on Japanese electronics goods o gold prices drop"
    )

with col2:
    # Selector de modelo (Por ahora enfocado en TF-IDF + Coseno, pero listo para BM25 y Jaccard)
    modelo_seleccionado = tf_web.selectbox(
        "Algoritmo de Recuperación:",
        ["TF-IDF + Similitud Coseno", "BM25", "Jaccard Binario"]
    )

# Slider para elegir cuántos documentos queremos ver en pantalla
top_k = tf_web.slider("Cantidad de resultados a recuperar (Top K):", min_value=1, max_value=20, value=5)

# --- EJECUCIÓN DE LA BÚSQUEDA ---
if query_usuario:
    tf_web.info(f"Procesando la consulta: *\"{query_usuario}\"* usando **{modelo_seleccionado}**")
    
    # Invocar el método correspondiente según la selección del usuario
    if modelo_seleccionado == "TF-IDF + Similitud Coseno":
        resultados = motor.buscar_coseno_tfidf(query_usuario, top_k=top_k)
    elif modelo_seleccionado == "BM25":
        resultados = motor.buscar_bm25(query_usuario, top_k=top_k)
    else:
        resultados = motor.buscar_jaccard(query_usuario, top_k=top_k)
        
    # Desplegar los resultados en pantalla
    if not resultados:
        tf_web.warning("⚠️ No se encontraron documentos relevantes para los términos ingresados.")
    else:
        tf_web.success(f"Se encontraron {len(resultados)} documentos relevantes.")
        
        # Iterar sobre el ranking e imprimirlos en tarjetas colapsables de Streamlit
        for rango, (doc_id, score) in enumerate(resultados, 1):
            # Obtener el texto completo original para mostrarlo
            texto_articulo = corpus_textos.get(doc_id, "Texto no encontrado en el corpus.")
            
            # Separar título y cuerpo (recordemos que los guardamos unidos por un salto de línea)
            lineas = texto_articulo.split('\n', 1)
            titulo = lineas[0] if lineas[0] else "Sin Título"
            cuerpo = lineas[1] if len(lineas) > 1 else ""
            
            # Crear un contenedor visual dinámico por cada noticia del ranking
            with tf_web.expander(f"🏅 Posición {rango} | Documento ID: {doc_id} (Score: {score:.4f}) — {titulo}"):
                tf_web.markdown(f"**Título Original:** {titulo}")
                tf_web.code(cuerpo, language="text")