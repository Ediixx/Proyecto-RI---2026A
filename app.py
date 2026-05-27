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

MODEL_COLORS = {
    "TF-IDF + Coseno": "#2563eb",
    "BM25": "#dc2626",
    "Jaccard": "#d97706",
    "Semántico": "#059669",
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
        .hit-snippet {
            padding: 0.85rem 1rem;
            border-left: 4px solid #ffb300;
            background: #fff8e7;
            color: #111827;
            border-radius: 10px;
            white-space: pre-wrap;
            line-height: 1.55;
            font-size: 0.95rem;
        }
        .hit-snippet mark {
            background: #fde68a;
            color: #111827;
            border-radius: 4px;
            padding: 0 0.15rem;
        }
        .doc-body {
            padding: 1rem;
            border: 1px solid #dbe3ef;
            border-radius: 10px;
            background: #f8fafc;
            color: #0f172a;
            line-height: 1.65;
            white-space: pre-wrap;
            font-size: 0.96rem;
        }
        .model-winner {
            padding: 0.75rem 1rem;
            background: #eef7ff;
            border: 1px solid #b9dcff;
            border-radius: 10px;
            margin-bottom: 0.75rem;
            color: #0f172a;
        }
        .model-winner * {
            color: #0f172a !important;
        }
        .model-card {
            border: 1px solid #d7deea;
            border-left: 7px solid var(--model-color);
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            background: #ffffff;
            color: #0f172a;
            margin-bottom: 0.75rem;
        }
        .model-card-title {
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .model-chip {
            display: inline-block;
            width: 11px;
            height: 11px;
            border-radius: 999px;
            background: var(--model-color);
            margin-right: 0.4rem;
            transform: translateY(1px);
        }
        .score-bar-row {
            margin-bottom: 0.62rem;
        }
        .score-bar-label {
            font-size: 0.9rem;
            color: #334155;
            margin-bottom: 0.2rem;
        }
        .score-bar-track {
            width: 100%;
            height: 11px;
            border-radius: 999px;
            background: #e8edf5;
            overflow: hidden;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: var(--model-color);
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
    """Ejecuta modelos y devuelve score crudo y normalizado de forma estable y legible."""
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
            df["score_normalizado"] = 0.5

        df["score_normalizado_pct"] = (df["score_normalizado"] * 100).round(1)
        df["color"] = df["modelo"].map(MODEL_COLORS).fillna("#64748b")
        df["rank_normalizado"] = (
            df["score_normalizado"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        orden_modelos = ["TF-IDF + Coseno", "BM25", "Jaccard", "Semántico"]
        df["_orden"] = df["modelo"].apply(lambda x: orden_modelos.index(x) if x in orden_modelos else 99)
        df = df.sort_values("_orden").drop(columns=["_orden"]).reset_index(drop=True)

    return df, resultados_por_modelo


def _render_score_bars(df, valor_columna, titulo, maximo, sufijo=""):
    st.caption(titulo)
    for _, fila in df.iterrows():
        modelo = str(fila["modelo"])
        color = fila.get("color", "#64748b")
        valor = float(fila[valor_columna])
        ancho = 0.0 if maximo <= 0 else (valor / maximo) * 100
        ancho = max(0.0, min(100.0, ancho))
        st.markdown(
            (
                '<div class="score-bar-row" style="--model-color: {color}">'
                '<div class="score-bar-label"><span class="model-chip"></span>{modelo}: <strong>{valor:.4f}{sufijo}</strong></div>'
                '<div class="score-bar-track"><div class="score-bar-fill" style="width: {ancho:.2f}%"></div></div>'
                "</div>"
            ).format(
                color=color,
                modelo=html.escape(modelo),
                valor=valor,
                sufijo=sufijo,
                ancho=ancho,
            ),
            unsafe_allow_html=True,
        )


def _render_model_cards(df):
    for _, fila in df.iterrows():
        modelo = str(fila["modelo"])
        color = fila.get("color", "#64748b")
        st.markdown(
            (
                '<div class="model-card" style="--model-color: {color}">'
                '<div class="model-card-title"><span class="model-chip"></span>{modelo}</div>'
                "Top 1: <strong>{doc}</strong><br>"
                "Score crudo: <strong>{crudo:.4f}</strong><br>"
                "Score normalizado: <strong>{norm:.1f}/100</strong><br>"
                "Recuperados: <strong>{rec}</strong> | Ranking normalizado: <strong>#{rank}</strong>"
                "</div>"
            ).format(
                color=color,
                modelo=html.escape(modelo),
                doc=html.escape(str(fila["doc_id_top1"])),
                crudo=float(fila["score_top1"]),
                norm=float(fila["score_normalizado_pct"]),
                rec=int(fila["docs_recuperados"]),
                rank=int(fila["rank_normalizado"]),
            ),
            unsafe_allow_html=True,
        )


def _render_grafico_final_comparativo(resultados_por_modelo):
    """Dibuja un solo gráfico con todos los modelos para comparar en el mismo plano."""
    filas = []
    orden = ["TF-IDF + Coseno", "BM25", "Jaccard", "Semántico"]

    for modelo in orden:
        lista_resultados = resultados_por_modelo.get(modelo, [])
        if not lista_resultados:
            continue

        max_modelo = max(float(score) for _, score in lista_resultados)
        max_modelo = max(max_modelo, 1e-12)

        for rank, (_, score) in enumerate(lista_resultados, 1):
            score_crudo = float(score)
            score_norm_0_100 = (score_crudo / max_modelo) * 100
            filas.append(
                {
                    "modelo": modelo,
                    "rank": rank,
                    "score_crudo": score_crudo,
                    "score_norm_0_100": score_norm_0_100,
                }
            )

    if not filas:
        st.caption("No hay datos suficientes para el gráfico comparativo final.")
        return

    df_chart = pd.DataFrame(filas)
    st.markdown("#### Gráfico final comparativo (todos los modelos en uno)")
    st.caption("Comparación por ranking usando score normalizado dentro de cada modelo (0-100).")

    escala_colores = [
        MODEL_COLORS["TF-IDF + Coseno"],
        MODEL_COLORS["BM25"],
        MODEL_COLORS["Jaccard"],
        MODEL_COLORS["Semántico"],
    ]

    st.vega_lite_chart(
        df_chart,
        {
            "mark": {"type": "line", "point": True, "strokeWidth": 3},
            "encoding": {
                "x": {"field": "rank", "type": "ordinal", "title": "Ranking (Top K)"},
                "y": {"field": "score_norm_0_100", "type": "quantitative", "title": "Score normalizado (0-100)", "scale": {"domain": [0, 100]}},
                "color": {
                    "field": "modelo",
                    "type": "nominal",
                    "scale": {
                        "domain": ["TF-IDF + Coseno", "BM25", "Jaccard", "Semántico"],
                        "range": escala_colores,
                    },
                    "legend": {"title": "Modelo"},
                },
                "tooltip": [
                    {"field": "modelo", "type": "nominal", "title": "Modelo"},
                    {"field": "rank", "type": "ordinal", "title": "Rank"},
                    {"field": "score_crudo", "type": "quantitative", "title": "Score crudo", "format": ".4f"},
                    {"field": "score_norm_0_100", "type": "quantitative", "title": "Score norm", "format": ".1f"},
                ],
            },
            "height": 320,
        },
        use_container_width=True,
    )


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
                mejor_fila = comparacion_df.sort_values(["score_normalizado", "score_top1"], ascending=False).iloc[0]
                st.markdown(
                    f'<div class="model-winner"><strong>Mejor modelo para esta consulta:</strong> {mejor_fila["modelo"]} '
                    f'| Documento top1: {mejor_fila["doc_id_top1"]} '
                    f'| Score normalizado: {mejor_fila["score_normalizado_pct"]:.1f}/100</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("#### Resumen visual por modelo")
                _render_model_cards(comparacion_df)

                st.dataframe(
                    comparacion_df[
                        [
                            "modelo",
                            "doc_id_top1",
                            "score_top1",
                            "score_normalizado_pct",
                            "docs_recuperados",
                            "rank_normalizado",
                        ]
                    ].rename(
                        columns={
                            "score_top1": "score_crudo_top1",
                            "score_normalizado_pct": "score_normalizado_0_100",
                            "rank_normalizado": "ranking_norm",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                grafico_comp_col1, grafico_comp_col2 = st.columns(2)
                with grafico_comp_col1:
                    _render_score_bars(
                        comparacion_df,
                        "score_normalizado_pct",
                        "Score top1 normalizado por modelo (0-100)",
                        maximo=100.0,
                        sufijo="",
                    )
                with grafico_comp_col2:
                    max_crudo = float(comparacion_df["score_top1"].max()) if not comparacion_df.empty else 0.0
                    _render_score_bars(
                        comparacion_df,
                        "score_top1",
                        "Score top1 crudo por modelo",
                        maximo=max_crudo,
                        sufijo="",
                    )

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

                    _render_grafico_final_comparativo(resultados_comparados)
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
                st.markdown(f'<div class="doc-body">{html.escape(cuerpo)}</div>', unsafe_allow_html=True)

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