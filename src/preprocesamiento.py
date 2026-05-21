# src/preprocesamiento.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Asegurar las descargas de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def limpiar_texto(texto):
    """
    Toma un texto string y aplica: minúsculas, remoción de caracteres especiales,
    tokenización, eliminación de stopwords y stemming (reducción a la raíz).
    """
    # 1. Normalización a minúsculas
    texto = str(texto).lower()
    
    # 2. Limpieza de saltos de línea y espacios dobles (común en Reuters)
    texto = re.sub(r'\s+', ' ', texto)
    
    # 3. Remover caracteres no alfabéticos (mantiene palabras puras)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    
    # 4. Tokenización
    tokens = word_tokenize(texto)
    
    # 5. Filtrado de Stopwords en inglés
    stop_words = set(stopwords.words('english'))
    tokens_filtrados = [word for word in tokens if word not in stop_words]
    
    # 6. Stemming (Opcional pero muy recomendado para TF-IDF y BM25)
    # Ej: "prices" y "pricing" se reducen a "price"
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(word) for word in tokens_filtrados]
    
    return tokens_stemmed