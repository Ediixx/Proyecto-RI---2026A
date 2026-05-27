# Sistema de Recuperación de Información (IR) - Proyecto RI 2026A

> **Aplicación principal en Streamlit para consultar Reuters-21578 con modelos clásicos y búsqueda semántica, mostrando rankings, comparaciones y evidencia textual.**

## Descripción General

La interfaz central del proyecto es [app.py](app.py), una aplicación **Streamlit** que permite:

- ejecutar consultas de texto libre sobre Reuters-21578;
- comparar modelos clásicos y semánticos desde una sola pantalla;
- ver rankings, scores, resúmenes y coincidencias resaltadas en el texto;
- reutilizar artefactos ya generados para arrancar más rápido.

El backend combina:

- **Modelos clásicos**: Jaccard, TF-IDF + coseno y BM25;
- **Búsqueda semántica**: embeddings con SentenceTransformer + ChromaDB;
- **Persistencia local**: índice invertido en disco, caché del corpus y base vectorial.

## Características principales de `app.py`

- Búsqueda por consulta libre en un solo panel.
- Selector de modelo: TF-IDF + Coseno, BM25, Jaccard o Semántico.
- Ranking de documentos con score y vista expandible.
- Fragmentos del documento con coincidencias resaltadas.
- Comparación entre modelos para la misma consulta con scores normalizados.
- Gráficas de resultados dentro de la interfaz.
- Caché persistente del corpus en `db/corpus_cache.pkl`.
- Reutilización del índice invertido en `db/indice_invertido.pkl`.
- Reutilización de la base vectorial en `db/vector_store/`.
- Panel lateral con estado del sistema y reconstrucción manual del índice.
- Soporte opcional para GPU o multiproceso en la indexación semántica.

---

## Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt`

---

## Instalación

### 1. Entrar al proyecto

```bash
cd Proyecto-RI---2026A
```

### 2. Crear y activar el entorno virtual

```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En macOS/Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar recursos de NLTK

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## Estructura del Proyecto

```text
Proyecto-RI---2026A/
├── app.py                      # Interfaz principal en Streamlit
├── cli.py                      # CLI secundaria para terminal
├── main.py                     # Script de demo rápida
├── requirements.txt            # Dependencias Python
├── README.md                   # Este archivo
│
├── data/                       # Corpus de datos
│   ├── ModApte_train.csv       # Entrenamiento
│   ├── ModApte_test.csv        # Prueba
│   └── qrels.json              # Ground truth para evaluación
│
├── db/                         # Artefactos persistentes
│   ├── corpus_cache.pkl        # Caché del corpus ya cargado
│   ├── indice_invertido.pkl    # Índice serializado
│   └── vector_store/           # Base vectorial ChromaDB
│
└── src/                        # Módulos principales
    ├── dataloader.py           # Cargador del corpus
    ├── preprocesamiento.py     # Limpieza y tokenización
    ├── indexacion.py           # Índice invertido
    ├── modelos_clasicos.py     # Jaccard, TF-IDF y BM25
    ├── semantica.py            # Embeddings + ChromaDB
    └── evaluacion.py           # Base para métricas de evaluación
```

---

## Uso principal: `app.py`

La forma recomendada de usar el proyecto es abrir la interfaz Streamlit:

```bash
streamlit run app.py
```

### Qué hace la aplicación

- carga o reutiliza el corpus desde `db/corpus_cache.pkl`;
- carga o reconstruye el índice clásico desde `db/indice_invertido.pkl`;
- reutiliza la base vectorial semántica si ya existe en `db/vector_store/`;
- permite elegir el modelo de recuperación;
- muestra documentos con resaltado de coincidencias;
- compara modelos para una misma consulta;
- dibuja gráficas con los scores obtenidos.

### Secciones de la interfaz

- **Buscar**: consulta libre, selección de modelo, ranking y comparación.
- **Documento**: consulta por ID de documento y muestra su contenido.
- **Modelos**: resumen de los modelos disponibles.
- **Sistema**: métricas de estado y artefactos usados.
- **Evaluación**: lectura de `qrels.json` y cálculo básico de métricas.

**Acceso local:** `http://localhost:8501`

---

## Flujo de `app.py`

1. Carga el corpus desde caché o desde los CSV de `data/`.
2. Carga el índice invertido serializado o lo reconstruye si no existe.
3. Inicializa el motor clásico con el índice.
4. Inicializa el motor semántico solo si se necesita.
5. Ejecuta la consulta y devuelve un ranking.
6. Resalta coincidencias, compara modelos y grafica scores.

---

## Modelos disponibles en la interfaz

### 1. Jaccard (Booleano)

$$\text{Jaccard}(Q, D) = \frac{|Q \cap D|}{|Q \cup D|}$$

- Ignora frecuencias y compara conjuntos de términos.
- Útil como baseline rápido.
- Implementado en: `src/modelos_clasicos.py::buscar_jaccard()`

### 2. TF-IDF + Similitud de Coseno

$$\text{TF}(t,d) = 1 + \log_{10}(freq(t,d))$$
$$\text{IDF}(t) = \log_{10}\left(\frac{N}{df(t)}\right)$$
$$\text{Score}(Q,D) = \cos(\vec{Q}, \vec{D})$$

- Modelo vectorial clásico.
- Balancea frecuencia local e importancia global.
- Implementado en: `src/modelos_clasicos.py::buscar_coseno_tfidf()`

### 3. BM25 (Okapi)

$$\text{Score}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{(k_1 + 1) \cdot f(q_i, D)}{k_1\left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right) + f(q_i, D)}$$

- Modelo probabilístico con normalización por longitud.
- Suele ser más robusto que TF-IDF en colecciones textuales.
- Implementado en: `src/modelos_clasicos.py::buscar_bm25()`

### 4. Búsqueda Semántica (Embeddings)

- **Modelo**: `all-MiniLM-L6-v2` (SentenceTransformer)
- **BD Vectorial**: ChromaDB persistente
- **Similitud**: coseno en espacio de embeddings
- **Dimensionalidad**: 384 dimensiones
- Implementado en: `src/semantica.py::MotorSemantico`

**Ventajas:**
- Captura significado semántico.
- Tolerancia a paráfrasis y sinónimos.
- No requiere coincidencia exacta de términos.

---

## Evaluación

La pestaña **Evaluación** de `app.py` permite leer `qrels.json` y calcular métricas básicas.

**Métricas a ampliar:**
- Precision@k
- Recall@k
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Comparativa más formal entre modelos

---

## Detalles Técnicos

### Índice Invertido

```python
indice = {
    'termino1': {doc_id1: freq1, doc_id2: freq2, ...},
    'termino2': {doc_id1: freq3, ...},
    ...
}
```

Persistencia: `db/indice_invertido.pkl`

### Base Vectorial (ChromaDB)

- Ubicación: `db/vector_store/`
- Recuperación: top-k por similitud coseno
- Eficiencia: búsqueda aproximada con HNSW

---

## Uso Programático

```python
from src.dataloader import cargar_corpus_completo
from src.modelos_clasicos import MotorClasico
from src.semantica import MotorSemantico

corpus, metadata = cargar_corpus_completo('data/')

# Motor clásico
motor_clasico = MotorClasico(corpus)
resultados_tfidf = motor_clasico.buscar_coseno_tfidf('tariffs electronics', top_k=5)

# Motor semántico
motor_semantico = MotorSemantico()
motor_semantico.indexar_corpus_vectorial(corpus)
resultados_semantic = motor_semantico.buscar_semantica('trade barriers on goods', top_k=5)
```

---

## Dependencias Clave

| Librería | Propósito |
|----------|-----------|
| pandas | Carga de CSV |
| nltk | Tokenización, stopwords y stemming |
| streamlit | Interfaz web |
| chromadb | Base de datos vectorial |
| sentence-transformers | Embeddings |

---

## Uso secundario: `cli.py`

La interfaz de línea de comandos sigue disponible para pruebas rápidas o para trabajar sin entorno gráfico.

```bash
python cli.py
```

Comandos habituales:

- `python cli.py init`
- `python cli.py search "tariffs electronics" --modelo tfidf --top 5`
- `python cli.py info`
- `python cli.py interactive`
- `python cli.py eval`
- `python cli.py doc 257`

## Preguntas Frecuentes

**P: ¿Cuál es el punto de entrada principal?**  
R: [app.py](app.py) es la interfaz principal.

**P: ¿La aplicación vuelve a cargar todo al iniciar?**  
R: Reutiliza `db/corpus_cache.pkl`, `db/indice_invertido.pkl` y `db/vector_store/` cuando ya existen.

**P: ¿El índice se construye automáticamente?**  
R: En `app.py` se reutiliza si existe; si no, se reconstruye.

**P: ¿Puedo agregar más modelos?**  
R: Sí, creando nuevas funciones en `src/modelos_clasicos.py` o una nueva clase semántica.

---

**Última actualización**: 2026-05-26  
**Estado**: ✅ Interfaz Streamlit centralizada y optimizada para reutilización de artefactos
