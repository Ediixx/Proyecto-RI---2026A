import json
import os

import pandas as pd
import streamlit as st

from src.dataloader import cargar_corpus_completo
from src.indexacion import IndiceInvertido
from src.modelos_clasicos import MotorClasico

try:
    from src.semantica import MotorSemantico
except Exception:
    MotorSemantico = None


RUTA_INDICE = "db/indice_invertido.pkl"
RUTA_DATA = "data"
RUTA_QRELS = "data/qrels.json"

MODEL_LABELS = {
    "TF-IDF + Coseno": "buscar_coseno_tfidf",
    "BM25": "buscar_bm25",
    "Jaccard": "buscar_jaccard",
}


st.set_page_config(
    page_title="Motor de Búsqueda Reuters-21578",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .app-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .app-subtitle {
            font-size: 1.05rem;
            color: #5b6573;
            margin-bottom: 1rem;
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


@st.cache_resource
def cargar_datos():
    return cargar_corpus_completo(RUTA_DATA)


@st.cache_resource
def cargar_motor_clasico():
    corpus, metadata_topics = cargar_datos()

    if os.path.exists(RUTA_INDICE) and os.path.getsize(RUTA_INDICE) > 0:
        indice = IndiceInvertido.cargar_de_disco(RUTA_INDICE)
        fuente_indice = "disco"
    else:
        indice = IndiceInvertido()
        indice.construir_indice(corpus)
        indice.guardar_en_disco(RUTA_INDICE)
        fuente_indice = "construido"

    return MotorClasico(indice), corpus, metadata_topics, indice, fuente_indice


@st.cache_resource
def cargar_motor_semantico():
    if MotorSemantico is None:
        raise RuntimeError("No se pudo importar el motor semántico.")

    corpus, _ = cargar_datos()
    motor_semantico = MotorSemantico()
    motor_semantico.indexar_corpus_vectorial(corpus)
    return motor_semantico


def _separar_texto(texto_articulo):
    lineas = str(texto_articulo).split("\n", 1)
    titulo = lineas[0] if lineas and lineas[0] else "Sin título"
    cuerpo = lineas[1] if len(lineas) > 1 else ""
    return titulo, cuerpo


def _normalizar_relevantes(valor):
    if isinstance(valor, dict):
        return {str(doc_id) for doc_id, relevancia in valor.items() if int(relevancia) > 0}
    if isinstance(valor, list):
        return {str(doc_id) for doc_id in valor}
    return set()


def _calcular_metricas(buscar_func, qrels, top_k):
    metricas = []
    for consulta, relevancia in qrels.items():
        relevantes = _normalizar_relevantes(relevancia)
        if not relevantes:
            continue

        resultados = buscar_func(str(consulta), top_k=top_k)
        recuperados = [str(doc_id) for doc_id, _ in resultados]

        tp = len(set(recuperados) & relevantes)
        precision = tp / len(recuperados) if recuperados else 0.0
        recall = tp / len(relevantes) if relevantes else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metricas.append(
            {
                "query": str(consulta)[:40],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return metricas


def _reconstruir_indice():
    corpus, _ = cargar_datos()
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
    motor_clasico, corpus_textos, metadata_topics, indice, fuente_indice = cargar_motor_clasico()
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

    ejecutar = st.button("Buscar", type="primary")

    if ejecutar and consulta.strip():
        with st.spinner("Procesando consulta..."):
            try:
                motor_semantico = cargar_motor_semantico() if modelo == "Semántico" else None
                resultados = _buscar(motor_clasico, motor_semantico, consulta, modelo, top_k)
            except Exception as error:
                st.error(f"La búsqueda falló: {error}")
                resultados = []

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
                        st.code(cuerpo, language="text")
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
        modelo_eval = st.selectbox("Modelo para evaluar", ["TF-IDF + Coseno", "BM25", "Jaccard"])
        top_k_eval = st.slider("Top K de evaluación", min_value=1, max_value=20, value=5, key="top_k_eval")

        if st.button("Ejecutar evaluación"):
            with st.spinner("Calculando métricas..."):
                metricas = _calcular_metricas(
                    lambda consulta, top_k=top_k_eval: _buscar(motor_clasico, None, consulta, modelo_eval, top_k),
                    qrels,
                    top_k_eval,
                )

            if not metricas:
                st.warning("No fue posible calcular métricas con los qrels cargados.")
            else:
                st.dataframe(metricas, use_container_width=True)

                precision_media = sum(item["precision"] for item in metricas) / len(metricas)
                recall_media = sum(item["recall"] for item in metricas) / len(metricas)
                f1_media = sum(item["f1"] for item in metricas) / len(metricas)

                m1, m2, m3 = st.columns(3)
                m1.metric("Precision media", f"{precision_media:.4f}")
                m2.metric("Recall medio", f"{recall_media:.4f}")
                m3.metric("F1 medio", f"{f1_media:.4f}")