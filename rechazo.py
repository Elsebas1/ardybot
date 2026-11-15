"""Ardy Bot - Etapa 3: decidir si una pregunta se puede responder con el documento o hay que declinar.

El clasificador mira 5 señales que salen del recuperador (no mira el texto de la pregunta directamente):
  bm25_max          mayor puntaje BM25 de la pregunta contra los fragmentos
  coseno_max        mayor similitud coseno (embeddings) contra los fragmentos
  n_desconocidos    términos de la pregunta que no aparecen en NINGÚN fragmento
  frac_desconocidos esa cantidad dividida entre el total de términos de la pregunta
  n_terminos        cuántos términos útiles tiene la pregunta
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recuperacion import tokenizar

NOMBRES = ["bm25_max", "coseno_max", "n_desconocidos", "frac_desconocidos", "n_terminos"]


def caracteristicas(consulta: str, bm25, denso) -> list[float]:
    mejor = bm25.buscar(consulta, 1)
    n_tok = len(tokenizar(consulta))
    n_desc = len(bm25.terminos_desconocidos(consulta))
    return [
        mejor[0][1] if mejor else 0.0,
        float(denso.puntajes(consulta).max()),
        n_desc,
        n_desc / n_tok if n_tok else 1.0,
        n_tok,
    ]


def nuevo_modelo():
    """Regresión logística con las señales estandarizadas (class_weight compensa que hay más preguntas válidas)."""
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))


class ClasificadorRechazo:
    """Se entrena con (pregunta, 1=responder / 0=declinar) y devuelve la probabilidad de poder responder."""

    def __init__(self, bm25, denso, umbral: float = 0.5):
        self.bm25, self.denso, self.umbral = bm25, denso, umbral
        self.modelo = None

    def entrenar(self, preguntas: list[str], etiquetas: list[int]):
        X = [caracteristicas(q, self.bm25, self.denso) for q in preguntas]
        self.modelo = nuevo_modelo().fit(X, etiquetas)
        return self

    def probabilidad(self, consulta: str) -> float:
        return float(self.modelo.predict_proba([caracteristicas(consulta, self.bm25, self.denso)])[0, 1])

    def puede_responder(self, consulta: str) -> tuple[bool, float]:
        p = self.probabilidad(consulta)
        return p >= self.umbral, p
