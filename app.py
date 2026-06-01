import json
import html
import os
import pickle
import re

import pandas as pd
import streamlit as st

from src.dataloader import cargar_corpus_completo
from src.indexacion import IndiceInvertido
from src.modelos_clasicos import MotorClasico
from src.evaluacion import evaluar_modelo, comparar_modelos, generar_reporte_comparativo

try:
    from src.semantica import MotorSemantico
except Exception:
    MotorSemantico = None


RUTA_INDICE = "db/indice_invertido.pkl"
RUTA_DATA = "data"
RUTA_QRELS = "data/qrels.json"
RUTA_CACHE_CORPUS = "db/corpus_cache.pkl"

ARCHIVOS_CORPUS = [
    os.path.join(RUTA_DATA, "ModApte_train.csv"),
    os.path.join(RUTA_DATA, "ModApte_test.csv"),
]

MODEL_LABELS = {
    "TF-IDF + Coseno": "buscar_coseno_tfidf",
    "BM25": "buscar_bm25",
    "Jaccard": "buscar_jaccard",
}

# Versiones normalizadas para evaluación comparativa
MODEL_LABELS_NORM = {
    "TF-IDF + Coseno": "buscar_coseno_tfidf_norm",
    "BM25": "buscar_bm25_norm",
    "Jaccard": "buscar_jaccard_norm",
}


def _version_cache_evaluacion():
    """Genera una versión simple para invalidar la caché cuando cambian los datos base."""
    partes = []
    for ruta in [RUTA_INDICE, RUTA_QRELS, *ARCHIVOS_CORPUS]:
        if os.path.exists(ruta):
            partes.append(str(os.path.getmtime(ruta)))
        else:
            partes.append("none")
    return "|".join(partes)


def _serializar_qrels_normalizados(qrels_normalizado):
    """Convierte los qrels normalizados a una cadena estable para cache."""
    qrels_ordenado = {
        str(consulta): sorted(str(doc_id) for doc_id in relevancia)
        for consulta, relevancia in sorted(qrels_normalizado.items(), key=lambda item: str(item[0]))
    }
    return json.dumps(qrels_ordenado, ensure_ascii=False, sort_keys=True)


def _deserializar_qrels_normalizados(qrels_serializados):
    """Reconstruye los qrels normalizados desde su representación serializada."""
    qrels = json.loads(qrels_serializados)
    return {consulta: set(doc_ids) for consulta, doc_ids in qrels.items()}


@st.cache_data(show_spinner=False)
def _evaluar_modelo_cacheado(modelo, qrels_serializados, top_k, version_cache):
    """Evalúa un modelo concreto reutilizando resultados cacheados cuando es posible."""
    del version_cache
    qrels_normalizado = _deserializar_qrels_normalizados(qrels_serializados)
    motor_clasico, _, _, _, _, _ = cargar_motor_clasico()

    if modelo == "Semántico":
        motor_semantico = cargar_motor_semantico()
        buscar_func = lambda q, top_k=top_k: motor_semantico.buscar_semantica(q, top_k)
    else:
        metodo_norm = MODEL_LABELS_NORM[modelo]
        buscar_func = lambda q, top_k=top_k, mt=metodo_norm: getattr(motor_clasico, mt)(q, top_k)

    return evaluar_modelo(buscar_func, qrels_normalizado, top_k=top_k)


@st.cache_data(show_spinner=False)
def _comparar_modelos_cacheado(qrels_serializados, top_k, version_cache):
    """Compara todos los modelos disponibles reutilizando resultados cacheados cuando es posible."""
    del version_cache
    qrels_normalizado = _deserializar_qrels_normalizados(qrels_serializados)
    motor_clasico, _, _, _, _, _ = cargar_motor_clasico()

    modelos_a_evaluar = {}
    for nombre in ["TF-IDF + Coseno", "BM25", "Jaccard"]:
        metodo_norm = MODEL_LABELS_NORM[nombre]
        modelos_a_evaluar[nombre] = lambda q, top_k=top_k, mt=metodo_norm: getattr(motor_clasico, mt)(q, top_k)

    if MotorSemantico is not None:
        motor_semantico = cargar_motor_semantico()
        modelos_a_evaluar["Semántico"] = lambda q, top_k=top_k: motor_semantico.buscar_semantica(q, top_k)

    return comparar_modelos(modelos_a_evaluar, qrels_normalizado, top_k=top_k)


st.set_page_config(
    page_title="Motor de Búsqueda Reuters-21578",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 4rem;
            padding-bottom: 2rem;
        }
        .app-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            margin-top: 1rem;
        }
        .app-subtitle {
            font-size: 1.05rem;
            color: #5b6573;
            margin-bottom: 1rem;
        }
            .hit-snippet {
                padding: 0.85rem 1rem;
                border-left: 4px solid #ffb300;
                background: #fff8e7;
                border-radius: 10px;
                white-space: pre-wrap;
                line-height: 1.55;
                font-size: 0.95rem;
            }
            .model-winner {
                padding: 0.75rem 1rem;
                background: #eef7ff;
                border: 1px solid #b9dcff;
                border-radius: 10px;
                margin-bottom: 0.75rem;
            }
    </style>
    """,
    unsafe_allow_html=True,
)


def _cargar_qrels(ruta_archivo):
    if not os.path.exists(ruta_archivo) or os.path.getsize(ruta_archivo) == 0:
        return {}

    try:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except Exception:
        return {}


def _obtener_estado_fuentes_corpus():
    """Retorna metadatos simples de los archivos fuente del corpus."""
    estado = {}
    for ruta in ARCHIVOS_CORPUS:
        if os.path.exists(ruta):
            estado[ruta] = os.path.getmtime(ruta)
        else:
            estado[ruta] = None
    return estado


def _cache_corpus_vigente(meta_cache):
    """Verifica si el cache en disco corresponde al estado actual de los CSV."""
    if not isinstance(meta_cache, dict):
        return False
    return meta_cache == _obtener_estado_fuentes_corpus()


def _guardar_cache_corpus(corpus, metadata_topics):
    os.makedirs(os.path.dirname(RUTA_CACHE_CORPUS), exist_ok=True)
    payload = {
        "fuentes": _obtener_estado_fuentes_corpus(),
        "corpus": corpus,
        "metadata_topics": metadata_topics,
    }
    with open(RUTA_CACHE_CORPUS, "wb") as archivo:
        pickle.dump(payload, archivo)


def _cargar_cache_corpus():
    if not os.path.exists(RUTA_CACHE_CORPUS) or os.path.getsize(RUTA_CACHE_CORPUS) == 0:
        return None

    try:
        with open(RUTA_CACHE_CORPUS, "rb") as archivo:
            payload = pickle.load(archivo)
        if _cache_corpus_vigente(payload.get("fuentes")):
            return payload.get("corpus"), payload.get("metadata_topics")
    except Exception:
        return None

    return None


@st.cache_resource
def cargar_datos():
    cache = _cargar_cache_corpus()
    if cache:
        corpus, metadata_topics = cache
        return corpus, metadata_topics, "cache_disco"

    corpus, metadata_topics = cargar_corpus_completo(RUTA_DATA)
    _guardar_cache_corpus(corpus, metadata_topics)
    return corpus, metadata_topics, "csv"


@st.cache_resource
def cargar_motor_clasico():
    corpus, metadata_topics, fuente_corpus = cargar_datos()

    if os.path.exists(RUTA_INDICE) and os.path.getsize(RUTA_INDICE) > 0:
        indice = IndiceInvertido.cargar_de_disco(RUTA_INDICE)
        fuente_indice = "disco"
    else:
        indice = IndiceInvertido()
        indice.construir_indice(corpus)
        indice.guardar_en_disco(RUTA_INDICE)
        fuente_indice = "construido"

    return MotorClasico(indice), corpus, metadata_topics, indice, fuente_indice, fuente_corpus


@st.cache_resource
def cargar_motor_semantico():
    if MotorSemantico is None:
        raise RuntimeError("No se pudo importar el motor semántico.")

    corpus, _, _ = cargar_datos()
    motor_semantico = MotorSemantico()
    motor_semantico.indexar_corpus_vectorial(corpus)
    return motor_semantico


def _separar_texto(texto_articulo):
    lineas = str(texto_articulo).split("\n", 1)
    titulo = lineas[0] if lineas and lineas[0] else "Sin título"
    cuerpo = lineas[1] if len(lineas) > 1 else ""
    return titulo, cuerpo


def _terminos_consulta(consulta):
    """Extrae términos de la consulta para resaltar coincidencias textuales."""
    return [
        termino.lower()
        for termino in re.findall(r"[A-Za-z0-9À-ÿ]+", consulta)
        if len(termino) > 1
    ]


def _resaltar_fragmento(texto, consulta, ventana=220):
    """Devuelve un fragmento de texto con las coincidencias resaltadas."""
    if not texto:
        return ""

    terminos = list(dict.fromkeys(_terminos_consulta(consulta)))
    texto_lower = texto.lower()

    posiciones = [texto_lower.find(termino) for termino in terminos if texto_lower.find(termino) != -1]
    if posiciones:
        centro = min(posiciones)
        inicio = max(0, centro - ventana)
        fin = min(len(texto), centro + ventana)
        fragmento = texto[inicio:fin]
    else:
        fragmento = texto[: ventana * 2]

    fragmento = html.escape(fragmento)

    for termino in sorted(set(terminos), key=len, reverse=True):
        patron = re.compile(rf"(?<!\w)({re.escape(termino)})(?!\w)", re.IGNORECASE)
        fragmento = patron.sub(r"<mark>\1</mark>", fragmento)

    return f'<div class="hit-snippet">{fragmento}</div>'


def _comparar_modelos(motor_clasico, motor_semantico, consulta, top_k):
    """Ejecuta todos los modelos disponibles y normaliza sus scores top para comparar la consulta."""
    modelos = ["TF-IDF + Coseno", "BM25", "Jaccard"]
    if motor_semantico is not None:
        modelos.append("Semántico")

    filas = []
    resultados_por_modelo = {}

    for modelo in modelos:
        resultados = _buscar(motor_clasico, motor_semantico if modelo == "Semántico" else None, consulta, modelo, top_k)
        resultados_por_modelo[modelo] = resultados
        if resultados:
            mejor_doc, mejor_score = resultados[0]
            filas.append(
                {
                    "modelo": modelo,
                    "doc_id_top1": str(mejor_doc),
                    "score_top1": float(mejor_score),
                    "docs_recuperados": len(resultados),
                }
            )
        else:
            filas.append(
                {
                    "modelo": modelo,
                    "doc_id_top1": "-",
                    "score_top1": 0.0,
                    "docs_recuperados": 0,
                }
            )

    df = pd.DataFrame(filas)
    if not df.empty:
        minimo = df["score_top1"].min()
        maximo = df["score_top1"].max()
        if maximo > minimo:
            df["score_normalizado"] = (df["score_top1"] - minimo) / (maximo - minimo)
        else:
            df["score_normalizado"] = 1.0
        df = df.sort_values(["score_normalizado", "score_top1"], ascending=False).reset_index(drop=True)

    return df, resultados_por_modelo


def _normalizar_relevantes(valor):
    if isinstance(valor, dict):
        return {str(doc_id) for doc_id, relevancia in valor.items() if int(relevancia) > 0}
    if isinstance(valor, list):
        return {str(doc_id) for doc_id in valor}
    return set()


def _calcular_metricas(buscar_func, qrels, top_k):
    """
    Versión heredada para compatibilidad. Usa evaluacion.evaluar_modelo.
    """
    qrels_normalizado = {}
    for consulta, relevancia in qrels.items():
        relevantes = _normalizar_relevantes(relevancia)
        if relevantes:
            qrels_normalizado[consulta] = relevantes
    
    resultado = evaluar_modelo(buscar_func, qrels_normalizado, top_k=top_k)
    return resultado['por_consulta']


def _reconstruir_indice():
    corpus, _, _ = cargar_datos()
    indice = IndiceInvertido()
    indice.construir_indice(corpus)
    indice.guardar_en_disco(RUTA_INDICE)
    st.cache_resource.clear()


def _buscar(motor_clasico, motor_semantico, consulta, modelo, top_k):
    if modelo == "Semántico":
        if motor_semantico is None:
            raise RuntimeError("El motor semántico no está disponible en este entorno.")
        return motor_semantico.buscar_semantica(consulta, top_k=top_k)

    metodo = MODEL_LABELS[modelo]
    buscar = getattr(motor_clasico, metodo)
    return buscar(consulta, top_k=top_k)


st.markdown('<div class="app-title">🔍 Sistema de Recuperación de Información</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Interfaz única en Streamlit para buscar, revisar documentos, comparar modelos y evaluar resultados sobre Reuters-21578.</div>',
    unsafe_allow_html=True,
)

try:
    motor_clasico, corpus_textos, metadata_topics, indice, fuente_indice, fuente_corpus = cargar_motor_clasico()
except Exception as error:
    st.error(f"No se pudo inicializar el backend: {error}")
    st.stop()

with st.sidebar:
    st.header("Controles")
    seccion = st.radio(
        "Ir a",
        ["Buscar", "Documento", "Modelos", "Sistema", "Evaluación"],
        index=0,
    )

    if st.button("Reconstruir índice clásico", use_container_width=True):
        with st.spinner("Recalculando índice invertido..."):
            _reconstruir_indice()
        st.success("Índice reconstruido. Recargando interfaz...")
        st.rerun()

    st.caption(f"Índice clásico: {fuente_indice}")
    st.caption(f"Corpus cargado desde: {fuente_corpus}")
    st.caption(f"Documentos cargados: {len(corpus_textos)}")
    st.caption(f"Términos únicos: {len(indice.indice)}")


if seccion == "Buscar":
    st.subheader("Búsqueda")
    col1, col2 = st.columns([3, 1])

    with col1:
        consulta = st.text_input(
            "Introduce tu consulta",
            placeholder="Ejemplo: tariffs on Japanese electronics goods",
        )

    with col2:
        modelo = st.selectbox(
            "Modelo",
            ["TF-IDF + Coseno", "BM25", "Jaccard", "Semántico"],
        )

    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
    comparar_modelos = st.checkbox("Comparar todos los modelos para esta consulta", value=True)

    ejecutar = st.button("Buscar", type="primary")

    if ejecutar and consulta.strip():
        with st.spinner("Procesando consulta..."):
            try:
                motor_semantico = None
                if modelo == "Semántico" or comparar_modelos:
                    try:
                        motor_semantico = cargar_motor_semantico()
                    except Exception as error_semantico:
                        if modelo == "Semántico":
                            raise
                        st.warning(f"No se pudo cargar el motor semántico para la comparación: {error_semantico}")
                resultados = _buscar(motor_clasico, motor_semantico, consulta, modelo, top_k)
                comparacion_df = None
                resultados_comparados = None
                if comparar_modelos:
                    comparacion_df, resultados_comparados = _comparar_modelos(motor_clasico, motor_semantico, consulta, top_k)
            except Exception as error:
                st.error(f"La búsqueda falló: {error}")
                resultados = []
                comparacion_df = None
                resultados_comparados = None

        if resultados:
            st.success(f"Se encontraron {len(resultados)} resultados.")
            resultados_df = []
            for posicion, (doc_id, score) in enumerate(resultados, 1):
                texto_articulo = corpus_textos.get(str(doc_id), corpus_textos.get(doc_id, "Texto no encontrado."))
                titulo, cuerpo = _separar_texto(texto_articulo)
                resultados_df.append(
                    {
                        "rank": posicion,
                        "doc_id": str(doc_id),
                        "score": float(score),
                        "titulo": titulo,
                    }
                )
                with st.expander(f"#{posicion} | Doc {doc_id} | Score {score:.4f} | {titulo}"):
                    st.markdown(f"**Título:** {titulo}")
                    st.markdown(f"**Documento ID:** {doc_id}")
                    if cuerpo:
                        st.markdown(_resaltar_fragmento(cuerpo, consulta), unsafe_allow_html=True)
                    else:
                        st.info("El documento no tiene cuerpo separado.")

            df_resultados = pd.DataFrame(resultados_df)
            st.markdown("### Gráficas de resultados")
            grafico_col1, grafico_col2 = st.columns(2)

            with grafico_col1:
                st.caption("Score por documento")
                st.bar_chart(df_resultados.set_index("doc_id")["score"])

            with grafico_col2:
                st.caption("Score por ranking")
                st.line_chart(df_resultados.set_index("rank")["score"])

            st.dataframe(
                df_resultados[["rank", "doc_id", "score", "titulo"]],
                use_container_width=True,
                hide_index=True,
            )

            if comparar_modelos and comparacion_df is not None and not comparacion_df.empty:
                st.markdown("### Comparación de modelos para la misma consulta")
                mejor_fila = comparacion_df.iloc[0]
                st.markdown(
                    f'<div class="model-winner"><strong>Mejor modelo para esta consulta:</strong> {mejor_fila["modelo"]} '
                    f'| Documento top1: {mejor_fila["doc_id_top1"]} '
                    f'| Score normalizado: {mejor_fila["score_normalizado"]:.4f}</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    comparacion_df[["modelo", "doc_id_top1", "score_top1", "score_normalizado", "docs_recuperados"]],
                    use_container_width=True,
                    hide_index=True,
                )

                grafico_comp_col1, grafico_comp_col2 = st.columns(2)
                with grafico_comp_col1:
                    st.caption("Score top1 normalizado por modelo")
                    st.bar_chart(comparacion_df.set_index("modelo")["score_normalizado"])
                with grafico_comp_col2:
                    st.caption("Score top1 crudo por modelo")
                    st.bar_chart(comparacion_df.set_index("modelo")["score_top1"])

                if resultados_comparados:
                    with st.expander("Ver resultados de cada modelo"):
                        for nombre_modelo, lista_resultados in resultados_comparados.items():
                            if not lista_resultados:
                                st.write(f"**{nombre_modelo}**: sin resultados")
                                continue
                            st.write(f"**{nombre_modelo}**")
                            mini_df = pd.DataFrame(
                                [
                                    {
                                        "rank": idx + 1,
                                        "doc_id": str(doc_id),
                                        "score": float(score),
                                    }
                                    for idx, (doc_id, score) in enumerate(lista_resultados)
                                ]
                            )
                            st.dataframe(mini_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No se encontraron documentos relevantes.")

elif seccion == "Documento":
    st.subheader("Consulta de documento")
    doc_id = st.text_input("Documento ID", placeholder="Ejemplo: 257")
    if st.button("Mostrar documento"):
        if not doc_id.strip():
            st.warning("Ingresa un ID de documento.")
        else:
            texto = corpus_textos.get(doc_id.strip(), corpus_textos.get(str(doc_id).strip(), None))
            if texto is None:
                st.error(f"No se encontró el documento {doc_id}.")
            else:
                titulo, cuerpo = _separar_texto(texto)
                st.markdown(f"### {titulo}")
                st.write(f"**ID:** {doc_id}")
                st.write(f"**Topics:** {', '.join(metadata_topics.get(str(doc_id), [])) or 'Sin topics'}")
                st.code(cuerpo, language="text")

elif seccion == "Modelos":
    st.subheader("Modelos disponibles")
    st.markdown(
        """
        - **TF-IDF + Coseno**: pondera términos por frecuencia e importancia global.
        - **BM25**: ranking probabilístico con normalización por longitud.
        - **Jaccard**: similitud binaria por intersección/union de términos.
        - **Semántico**: compara embeddings y recupera por significado.
        """
    )

    st.code(
        """
TF-IDF + Coseno -> MotorClasico.buscar_coseno_tfidf()
BM25            -> MotorClasico.buscar_bm25()
Jaccard         -> MotorClasico.buscar_jaccard()
Semántico       -> MotorSemantico.buscar_semantica()
        """.strip(),
        language="text",
    )

elif seccion == "Sistema":
    st.subheader("Estado del sistema")
    col1, col2, col3 = st.columns(3)

    col1.metric("Documentos", len(corpus_textos))
    col2.metric("Términos únicos", len(indice.indice))
    col3.metric("Longitud media", f"{motor_clasico.avgdl:.2f}" if motor_clasico.avgdl else "0.00")

    st.markdown("### Archivos principales")
    st.write(f"- Índice clásico: {RUTA_INDICE}")
    st.write(f"- Datos: {RUTA_DATA}")
    st.write(f"- Qrels: {RUTA_QRELS}")

    st.markdown("### Metadata")
    topics_contados = sum(1 for topics in metadata_topics.values() if topics)
    st.write(f"- Documentos con topics: {topics_contados}")
    st.write(f"- Documentos sin topics: {len(metadata_topics) - topics_contados}")

elif seccion == "Evaluación":
    st.subheader("Evaluación")
    st.write("Usa este panel para comparar resultados con un archivo QREL en formato JSON.")

    qrels_cargados = _cargar_qrels(RUTA_QRELS)
    archivo_qrels = st.file_uploader("Sube un archivo JSON de qrels", type=["json"])

    qrels = qrels_cargados
    if archivo_qrels is not None:
        try:
            qrels = json.load(archivo_qrels)
        except Exception:
            st.error("El archivo JSON subido no es válido.")

    if not qrels:
        st.warning("No hay qrels cargados. `data/qrels.json` está vacío o no contiene datos.")
    else:
        opciones_modelos = ["TF-IDF + Coseno", "BM25", "Jaccard"]
        if MotorSemantico is not None:
            opciones_modelos.append("Semántico")
        
        modelo_eval = st.selectbox("Modelo para evaluar", opciones_modelos)
        top_k_eval = st.slider("Top K de evaluación", min_value=1, max_value=20, value=5, key="top_k_eval")
        
        comparar_todos = st.checkbox("Comparar todos los modelos", value=False)

        if st.button("Ejecutar evaluación"):
            with st.spinner("Calculando métricas..."):
                # Normalizar qrels
                qrels_normalizado = {}
                for consulta, relevancia in qrels.items():
                    relevantes = _normalizar_relevantes(relevancia)
                    if relevantes:
                        qrels_normalizado[consulta] = relevantes

                qrels_serializados = _serializar_qrels_normalizados(qrels_normalizado)
                version_cache = _version_cache_evaluacion()
                
                if comparar_todos:
                    cache_key = f"eval_comparacion::{top_k_eval}::{version_cache}::{qrels_serializados}"
                    if cache_key in st.session_state:
                        resultados_comparacion = st.session_state[cache_key]
                        st.info("Resultados cargados desde caché de la sesión.")
                    else:
                        # Comparar todos los modelos disponibles
                        modelos_a_evaluar = {}

                        # Modelos clásicos con versiones NORMALIZADAS para comparación justa
                        for nombre in ["TF-IDF + Coseno", "BM25", "Jaccard"]:
                            metodo_norm = MODEL_LABELS_NORM[nombre]
                            buscar_func = lambda q, top_k=top_k_eval, mt=metodo_norm: getattr(
                                motor_clasico, mt
                            )(q, top_k)
                            modelos_a_evaluar[nombre] = buscar_func

                        # Modelo semántico (ya retorna scores normalizados)
                        if MotorSemantico is not None:
                            try:
                                motor_semantico = cargar_motor_semantico()
                                buscar_semantico = lambda q, top_k=top_k_eval: motor_semantico.buscar_semantica(q, top_k)
                                modelos_a_evaluar["Semántico"] = buscar_semantico
                            except Exception as e:
                                st.warning(f"No se pudo cargar modelo semántico: {e}")

                        # Evaluar todos los modelos
                        progress_bar_comp = st.progress(0, text="Inicializando comparación...")

                        def actualizar_progreso_comp(actual, total, mensaje):
                            progreso = actual / total if total > 0 else 0
                            progress_bar_comp.progress(progreso, text=mensaje)

                        resultados_comparacion = comparar_modelos(
                            modelos_a_evaluar,
                            qrels_normalizado,
                            top_k=top_k_eval,
                            progress_callback=actualizar_progreso_comp
                        )
                        progress_bar_comp.empty()  # Limpiar barra al terminar
                        st.session_state[cache_key] = resultados_comparacion
                    
                    # Mostrar resultados tabulares
                    st.markdown("### Comparación de Modelos")
                    
                    # Crear tabla comparativa
                    filas_comparacion = []
                    for modelo, resultado in resultados_comparacion.items():
                        if resultado is None:
                            filas_comparacion.append({
                                'Modelo': modelo,
                                'Precision': 'N/D',
                                'Recall': 'N/D',
                                'F1': 'N/D',
                                'MAP': 'N/D',
                                'Consultas': 0,
                                'Estado': 'No disponible o con error'
                            })
                            continue

                        agg = resultado['agregadas']
                        filas_comparacion.append({
                            'Modelo': modelo,
                            'Precision': f"{agg['precision_media']:.4f}",
                            'Recall': f"{agg['recall_media']:.4f}",
                            'F1': f"{agg['f1_media']:.4f}",
                            'MAP': f"{agg['map']:.4f}",
                            'Consultas': agg['num_consultas'],
                            'Estado': 'OK'
                        })
                    
                    df_comparacion = pd.DataFrame(filas_comparacion)
                    st.dataframe(df_comparacion, use_container_width=True, hide_index=True)
                    
                    # Gráficos comparativos
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        datos_map = []
                        for modelo, resultado in resultados_comparacion.items():
                            if resultado:
                                datos_map.append({
                                    'Modelo': modelo,
                                    'MAP': resultado['agregadas']['map']
                                })
                        if datos_map:
                            df_map = pd.DataFrame(datos_map).set_index('Modelo')
                            st.bar_chart(df_map)
                            st.caption("MAP por modelo")
                    
                    with col2:
                        datos_f1 = []
                        for modelo, resultado in resultados_comparacion.items():
                            if resultado:
                                datos_f1.append({
                                    'Modelo': modelo,
                                    'F1': resultado['agregadas']['f1_media']
                                })
                        if datos_f1:
                            df_f1 = pd.DataFrame(datos_f1).set_index('Modelo')
                            st.bar_chart(df_f1)
                            st.caption("F1 Promedio por modelo")
                    
                    # Detalles por consulta para modelo seleccionado
                    if modelo_eval in resultados_comparacion and resultados_comparacion[modelo_eval]:
                        with st.expander(f"📊 Detalle por consulta - {modelo_eval}"):
                            df_detalle = pd.DataFrame(resultados_comparacion[modelo_eval]['por_consulta'])
                            st.dataframe(df_detalle, use_container_width=True, hide_index=True)
                    elif comparar_todos:
                        st.info("Algún modelo no devolvió resultados. Revisa la columna 'Estado' en la tabla de comparación.")
                
                else:
                    # Evaluar un solo modelo con versión NORMALIZADA para comparación justa
                    if modelo_eval == "Semántico":
                        try:
                            motor_semantico = cargar_motor_semantico()
                            buscar_func = lambda q, top_k=top_k_eval: motor_semantico.buscar_semantica(q, top_k)
                        except Exception as e:
                            st.error(f"No se pudo cargar modelo semántico: {e}")
                            st.stop()
                    else:
                        # Usar versión normalizada para modelos clásicos
                        metodo_norm = MODEL_LABELS_NORM[modelo_eval]
                        buscar_func = lambda q, top_k=top_k_eval, mt=metodo_norm: getattr(
                            motor_clasico, mt
                        )(q, top_k)

                    cache_key = f"eval_modelo::{modelo_eval}::{top_k_eval}::{version_cache}::{qrels_serializados}"
                    if cache_key in st.session_state:
                        resultado = st.session_state[cache_key]
                        st.info("Resultados cargados desde caché de la sesión.")
                    else:
                        # Crear barra de progreso
                        progress_bar = st.progress(0, text="Iniciando evaluación...")

                        def actualizar_progreso(actual, total, mensaje):
                            progreso = actual / total
                            progress_bar.progress(progreso, text=mensaje)

                        resultado = evaluar_modelo(
                            buscar_func,
                            qrels_normalizado,
                            top_k=top_k_eval,
                            progress_callback=actualizar_progreso
                        )
                        progress_bar.empty()  # Limpiar barra al terminar
                        st.session_state[cache_key] = resultado
                    
                    metricas = resultado['por_consulta']
                    agg = resultado['agregadas']
                    
                    if not metricas:
                        st.warning("No fue posible calcular métricas con los qrels cargados.")
                    else:
                        st.markdown(f"### Evaluación: {modelo_eval} (scores normalizados)")
                        
                        # Tabla de métricas por consulta
                        df_metricas = pd.DataFrame(metricas)
                        st.dataframe(df_metricas, use_container_width=True, hide_index=True)
                        
                        # Métricas agregadas
                        st.markdown("### Métricas Agregadas")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Precision media", f"{agg['precision_media']:.4f}")
                        m2.metric("Recall medio", f"{agg['recall_media']:.4f}")
                        m3.metric("F1 medio", f"{agg['f1_media']:.4f}")
                        m4.metric("MAP", f"{agg['map']:.4f}")
