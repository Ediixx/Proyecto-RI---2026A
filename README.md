# Sistema de Recuperación de Información (IR) - Proyecto RI 2026A

> **Motor de búsqueda académico que compara modelos clásicos de recuperación de información con búsqueda semántica moderna usando el corpus Reuters-21578 (ModApte)**

## 📋 Descripción General

Este proyecto implementa una plataforma educativa de **Recuperación de Información** que permite experimentar con diferentes algoritmos de ranking y búsqueda sobre un corpus de noticias financieras. Incluye:

- **Modelos Clásicos**: Jaccard (booleano), TF-IDF (vectorial) y BM25 (probabilístico)
- **Búsqueda Semántica**: Embeddings con SentenceTransformer + ChromaDB
- **Interfaz Multiple**: CLI, Web UI (Streamlit), y Script standalone
- **Corpus**: Reuters-21578 (ModApte) con ~10,788 documentos categorizados

---

## ✨ Características

- ✅ Tres algoritmos de ranking clásicos (Jaccard, TF-IDF, BM25)
- ✅ Búsqueda semántica con embeddings BERT
- ✅ Índice invertido persistente
- ✅ Base vectorial ChromaDB para búsqueda semántica
- ✅ Interfaz web interactiva con Streamlit
- ✅ CLI para experimentación rápida
- ✅ Pipeline de preprocesamiento (tokenización, stopwords, stemming)
- ✅ Corpus categorizado con topics/tópicos

---

## 📦 Requisitos

- **Python 3.8+**
- Librerías especificadas en `requirements.txt`

---

## 🚀 Instalación

### 1. Clonar o descargar el proyecto

```bash
cd Proyecto-RI---2026A
```

### 2. Crear un entorno virtual (recomendado)

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar recursos de NLTK (primera vez)

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 📂 Estructura del Proyecto

```
Proyecto-RI---2026A/
├── app.py                      # Aplicación web (Streamlit)
├── cli.py                      # Interfaz de línea de comandos
├── main.py                     # Script de demostración
├── requirements.txt            # Dependencias Python
├── README.md                   # Este archivo
│
├── data/                       # Corpus de datos
│   ├── ModApte_train.csv       # Documentos de entrenamiento
│   ├── ModApte_test.csv        # Documentos de prueba
│   └── qrels.json              # Ground truth para evaluación
│
├── db/                         # Almacenamiento persistente
│   ├── indice_invertido.pkl    # Índice serializado
│   └── vector_store/           # Base vectorial ChromaDB
│
└── src/                        # Módulos principales
    ├── __init__.py
    ├── dataloader.py           # Cargador del corpus
    ├── preprocesamiento.py     # Pipeline de limpieza de texto
    ├── indexacion.py           # Construcción de índices
    ├── modelos_clasicos.py     # Algoritmos: Jaccard, TF-IDF, BM25
    ├── semantica.py            # Búsqueda con embeddings
    └── evaluacion.py           # Métricas de evaluación (framework)
```

---

## 🎯 Uso

### Opción 1: Aplicación Web (Streamlit)

Interfaz interactiva con UI web:

```bash
streamlit run app.py
```

**Funcionalidades:**
- Motor de búsqueda TF-IDF
- Índice persistente de `db/indice_invertido.pkl`
- Resultados con scores de similitud
- Layout optimizado para escritorio

**Acceso:** http://localhost:8501

---

### Opción 2: Interfaz CLI

Experimentación rápida desde terminal:

```bash
python cli.py
```

**Modelos disponibles:**
- `tfidf`: Búsqueda con TF-IDF + Coseno
- `booleano`: Similitud Jaccard (intersection/union)
- `bm25`: Modelo probabilístico Okapi BM25

**Ejemplo en el código:**
```python
from cli import RICLI
ri = RICLI()
ri.buscar(modelo='tfidf', consulta='tariffs Japanese electronics')
```

---

### Opción 3: Script de Demostración

Ejecución rápida con query predefinido:

```bash
python main.py
```

**Qué hace:**
- Carga corpus Reuters
- Busca: "tariffs on Japanese electronics goods"
- Muestra Top 5 resultados con scores TF-IDF

---

## 🧠 Modelos Implementados

### 1. **Jaccard (Booleano)**
$$\text{Jaccard}(Q, D) = \frac{|Q \cap D|}{|Q \cup D|}$$

- Ignora frecuencias (vectores binarios)
- Rápido pero menos preciso
- Implementado en: `modelos_clasicos.py::MotorClasico.buscar_jaccard()`

---

### 2. **TF-IDF + Similitud de Coseno**
$$\text{TF}(t,d) = 1 + \log_{10}(freq(t,d))$$
$$\text{IDF}(t) = \log_{10}\left(\frac{N}{df(t)}\right)$$
$$\text{Score}(Q,D) = \cos(\vec{Q}, \vec{D})$$

- Clásico de recuperación de información
- Balancea frecuencia de término e importancia global
- Implementado en: `modelos_clasicos.py::MotorClasico.buscar_coseno_tfidf()`

---

### 3. **BM25 (Okapi)**
$$\text{Score}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{(k_1 + 1) \cdot f(q_i, D)}{k_1\left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right) + f(q_i, D)}$$

- Modelo probabilístico mejorado
- Considera longitud del documento
- Parámetros: `k1=1.5` (saturación), `b=0.75` (normalización de longitud)
- Implementado en: `modelos_clasicos.py::MotorClasico.buscar_bm25()`

---

### 4. **Búsqueda Semántica (Embeddings)**

- **Modelo**: `all-MiniLM-L6-v2` (SentenceTransformer)
- **BD Vectorial**: ChromaDB (persistente)
- **Similitud**: Coseno en espacio de embeddings
- **Dimensionalidad**: 384 dimensiones
- Implementado en: `semantica.py::MotorSemantico`

**Ventajas:**
- Captura significado semántico
- Tolerancia a paráfrasis y sinónimos
- No requiere coincidencia exacta de términos

---

## 📊 Corpus de Datos

**Nombre**: Reuters-21578 (ModApte split)

**Características:**
- **Documentos**: ~10,788 documentos
- **Idioma**: Inglés
- **Dominio**: Noticias financieras
- **Estructura**: Título + Cuerpo + Tópicos
- **Formato**: CSV (train/test split)

**Tópicos Principales**: earn, acq (adquisiciones), money-fx, grain, crude, trade, interest, ship, corn, wheat, etc.

**Archivo Ground Truth**: `qrels.json` - Relevancia esperada para evaluación

---

## 🔄 Pipeline de Preprocesamiento

Implementado en `src/preprocesamiento.py`:

1. **Conversión a minúsculas**
2. **Normalización de espacios**
3. **Eliminación de caracteres especiales** (solo letras y dígitos)
4. **Tokenización** (NLTK punkt)
5. **Eliminación de stopwords** (inglés: "the", "a", "is", etc.)
6. **Stemming** (Porter Stemmer: "running" → "run")

```python
from src.preprocesamiento import limpiar_texto

texto = "The prices are RISING!"
limpio = limpiar_texto(texto)
# Resultado: ['price', 'rise']
```

---

## 📈 Evaluación

**Estado**: Framework disponible en `src/evaluacion.py`

**Métricas a implementar:**
- Precision@k (P@5, P@10)
- Recall@k
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Comparativa entre modelos

**Ground Truth**: `data/qrels.json`

---

## 🛠️ Detalles Técnicos

### Índice Invertido

Estructura:
```python
indice = {
    'termino1': {doc_id1: freq1, doc_id2: freq2, ...},
    'termino2': {doc_id1: freq3, ...},
    ...
}
```

**Persistencia**: Serialización con pickle en `db/indice_invertido.pkl`

---

### Base Vectorial (ChromaDB)

- **Ubicación**: `db/vector_store/`
- **Capacidad**: Todos los documentos del corpus
- **Recuperación**: Top-k por similitud coseno
- **Eficiencia**: Búsqueda O(1) con índices HNSW

---

## 📝 Ejemplo de Uso Programático

```python
from src.dataloader import cargar_corpus_completo
from src.modelos_clasicos import MotorClasico
from src.semantica import MotorSemantico

# 1. Cargar datos
corpus, metadata = cargar_corpus_completo('data/')

# 2. Crear motor clásico
motor_clasico = MotorClasico(corpus)

# 3. Búsqueda TF-IDF
resultados_tfidf = motor_clasico.buscar_coseno_tfidf(
    consulta="tariffs electronics",
    top_k=5
)

for doc_id, score in resultados_tfidf:
    print(f"Doc {doc_id}: {score:.4f}")

# 4. Búsqueda BM25
resultados_bm25 = motor_clasico.buscar_bm25(
    consulta="tariffs electronics",
    top_k=5
)

# 5. Búsqueda Semántica
motor_semantico = MotorSemantico()
motor_semantico.indexar_corpus_vectorial(corpus)
resultados_semantic = motor_semantico.buscar_semantica(
    consulta="trade barriers on goods",
    top_k=5
)
```

---

## 📚 Dependencias Clave

| Librería | Versión | Propósito |
|----------|---------|----------|
| pandas | Latest | Manipulación de datos (CSV) |
| nltk | Latest | NLP: tokenización, stopwords, stemming |
| streamlit | Latest | Interfaz web interactiva |
| chromadb | Latest | Base de datos vectorial |
| sentence-transformers | Latest | Generación de embeddings BERT |
| scikit-learn | Latest | Utilities (si se usa) |

---

## 🎓 Objetivos Educativos

✅ Comprender cómo funcionan algoritmos de ranking clásicos  
✅ Implementar estructura de datos (índice invertido)  
✅ Aprender normalización de texto y preprocesamiento  
✅ Explorar representaciones semánticas (embeddings)  
✅ Comparar enfoques: frecuencial vs. semántico  
✅ Evaluar sistemas de información  

---

## 🤝 Contribuciones

Áreas pendientes o mejoras:

- [ ] Completar `src/evaluacion.py` con métricas
- [ ] Optimizar búsqueda vectorial
- [ ] Agregar más modelos (BM25F, QL, etc.)
- [ ] Interfaz CLI mejorada
- [ ] Caché de consultas frecuentes
- [ ] Soporte para múltiples idiomas

---

## 📄 Licencia

Proyecto académico - RI 2026A

---

## 👤 Autor

Desarrollado como proyecto educativo de Recuperación de Información (2026A)

---

## ❓ Preguntas Frecuentes

**P: ¿Por qué el archivo `evaluacion.py` está vacío?**  
R: Es un framework listo para implementar. Se puede usar `qrels.json` como ground truth.

**P: ¿Puedo agregar más modelos?**  
R: Sí, crea nuevas clases heredando de `MotorClasico` o `MotorSemantico`.

**P: ¿El índice se construye automáticamente?**  
R: En `app.py` se carga desde `db/indice_invertido.pkl` si existe. Úsalo en scripts para regenerar.

**P: ¿Cómo comparar modelos?**  
R: Usa `evaluacion.py` para calcular Precision, Recall, MAP sobre `qrels.json`.

---

**Última actualización**: 2026-05-25  
**Estado**: ✅ Funcional con framework de evaluación pendiente