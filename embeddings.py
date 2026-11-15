"""Ardy Bot - Etapa 2: recuperación semántica (embeddings) y recuperación híbrida (BM25 + embeddings).

Idea: un "encoder" convierte cada fragmento y cada pregunta en un vector; fragmentos de significado
parecido quedan con vectores parecidos. Se ordena por similitud coseno.
"""
import numpy as np

MODELO_POR_DEFECTO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # multilingüe, incluye español


def encoder_sbert(nombre: str = MODELO_POR_DEFECTO):
    """Encoder real. Requiere: python3 -m pip install sentence-transformers (la 1.ª vez descarga el modelo)."""
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(nombre)
    return lambda textos: modelo.encode(list(textos), normalize_embeddings=True, show_progress_bar=False)


def encoder_simulado(textos_corpus):
    """SOLO PARA PROBAR EL FLUJO sin descargar modelos: vectores TF-IDF de caracteres.
    NO es semántico: no sirve para reportar resultados."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode").fit(textos_corpus)

    def enc(textos):
        m = vec.transform(list(textos)).toarray()
        norma = np.linalg.norm(m, axis=1, keepdims=True)
        norma[norma == 0] = 1
        return m / norma
    return enc


class DenseRetriever:
    """Recuperación semántica: similitud coseno entre el vector de la pregunta y el de cada fragmento."""

    def __init__(self, textos: list[str], encoder):
        self.encoder = encoder
        self.matriz = np.asarray(encoder(textos))  # una fila por fragmento (ya normalizada)

    def puntajes(self, consulta: str) -> np.ndarray:
        return self.matriz @ np.asarray(self.encoder([consulta]))[0]

    def buscar(self, consulta: str, k: int = 3) -> list[tuple[int, float]]:
        s = self.puntajes(consulta)
        return [(int(i), float(s[i])) for i in np.argsort(-s)[:k]]


def _minmax(v: np.ndarray) -> np.ndarray:
    """Lleva los puntajes a la escala 0-1 para poder sumar BM25 y coseno."""
    rango = v.max() - v.min()
    return (v - v.min()) / rango if rango > 0 else np.zeros_like(v)


class HybridRetriever:
    """Híbrido: suma ponderada alpha*BM25 + (1-alpha)*coseno, ambos normalizados a 0-1.
    alpha=1 equivale a solo BM25; alpha=0, a solo embeddings."""

    def __init__(self, bm25, denso: DenseRetriever, alpha: float = 0.5):
        self.bm25, self.denso, self.alpha = bm25, denso, alpha

    def puntajes_bm25(self, consulta: str) -> np.ndarray:
        s = np.zeros(self.bm25.N)
        for i, sc in self.bm25.buscar(consulta, k=self.bm25.N):  # BM25 solo devuelve los que puntúan > 0
            s[i] = sc
        return s

    def buscar(self, consulta: str, k: int = 3) -> list[tuple[int, float]]:
        h = self.alpha * _minmax(self.puntajes_bm25(consulta)) + (1 - self.alpha) * _minmax(self.denso.puntajes(consulta))
        return [(int(i), float(h[i])) for i in np.argsort(-h)[:k]]
