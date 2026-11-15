"""Ardy Bot - Etapa 1: ingesta del documento y recuperación con BM25.

Pasos:
  1. cargar_chunks(): lee el .md validado y lo parte en fragmentos (chunks).
  2. tokenizar():     convierte texto en términos, conservando códigos como "1-11" o "CEP-11".
  3. BM25:            ordena los fragmentos según su relevancia para una pregunta.
"""
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

# Palabras que no ayudan a distinguir un fragmento de otro.
STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "de", "del", "al", "en", "y", "a", "que", "qué",
    "es", "se", "me", "por", "con", "para", "donde", "dónde", "queda", "esta", "está",
    "cual", "cuál", "como", "cómo", "hay", "su", "sus", "lo", "o",
}


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes: 'Salón' -> 'salon'."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


# Un término es letras/números, opcionalmente unidos por guiones (ej. cep-11, 1-11).
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP_NORM = {normalizar(s) for s in STOPWORDS}


def tokenizar(texto: str) -> list[str]:
    tokens = []
    for t in _TOKEN.findall(normalizar(texto)):
        if t in _STOP_NORM:
            continue
        tokens.append(t)
        # 'cep-11' también aporta 'cep', para que la pregunta "salones del CEP" encuentre el código.
        if "-" in t:
            tokens.extend(p for p in t.split("-") if p.isalpha() and len(p) >= 2)
    return tokens


# ----------------------------------------------------------------------------- chunks
def _celdas(linea: str) -> list[str]:
    return [c.strip() for c in linea.strip().strip("|").split("|")]


def cargar_chunks(ruta_md: str) -> list[dict]:
    """Devuelve una lista de {'id','texto','fuente'} a partir del documento de mapas."""
    chunks: list[dict] = []

    def agregar(texto: str, fuente: str):
        chunks.append({"id": f"c{len(chunks) + 1:03d}", "texto": texto, "fuente": fuente})

    seccion = ""
    for linea in Path(ruta_md).read_text(encoding="utf-8").splitlines():
        if linea.startswith("## "):
            seccion = linea[3:].strip()[0]  # '1', '2', '3' o '4'
            continue

        # Sección 1: tabla de salones -> un chunk por salón y otro por bloque/piso.
        if seccion == "1" and linea.startswith("|"):
            c = _celdas(linea)
            if c[0] in ("Marcador",) or set(c[0]) <= {"-"}:
                continue
            _, edificio, piso, salones = c
            piso_txt = "piso no indicado en el mapa" if piso.startswith("no indicado") else piso
            lugar = f"el {edificio}" if edificio.startswith("Bloque") else edificio
            agregar(f"Salones de {edificio}, {piso_txt}: {salones}.", "Mapa de salones 2022")
            for item in (s.strip() for s in salones.split(",")):
                m = re.match(r"([A-Za-z0-9\-]+)\s*(?:\((.*)\))?$", item)
                codigo, nota = m.group(1), m.group(2)
                nota_txt = f" ({nota})" if nota else ""
                agregar(f"Salón {codigo}{nota_txt}: queda en {lugar}, {piso_txt}.", "Mapa de salones 2022")

        # Sección 2: tabla de puntos numerados + listas de entradas, parqueaderos, etc.
        elif seccion == "2":
            if linea.startswith("|"):
                c = _celdas(linea)
                if c[0] == "Punto" or set(c[0]) <= {"-"}:
                    continue
                punto = c[0].replace("(?)", "").strip()
                agregar(f"Punto {punto} del Mapa Institucional: {c[1]}.", "Mapa Institucional 2025")
            else:
                m = re.match(r"^(Entradas|Parqueaderos|Auditorios|Cafeterías|Lugares de interés):\s*(.*)\.$", linea)
                if m:
                    singular = {"Entradas": "Entrada", "Parqueaderos": "Parqueadero", "Auditorios": "Auditorio",
                                "Cafeterías": "Cafetería", "Lugares de interés": "Lugar de interés"}[m.group(1)]
                    for it in m.group(2).split(";"):
                        mm = re.match(r"\s*([A-Z]\d)\s+(.*)", it)
                        agregar(f"{singular} {mm.group(1)}: {mm.group(2).strip()} (Mapa Institucional).",
                                "Mapa Institucional 2025")

        # Sección 3: equivalencias y ubicaciones (cada viñeta es un chunk).
        elif seccion == "3" and linea.startswith("- "):
            agregar(linea[2:].strip(), "Equivalencias entre mapas")
    return chunks


# ------------------------------------------------------------------------------ BM25
class BM25:
    """BM25 Okapi: puntúa más alto a los fragmentos que contienen los términos raros de la pregunta."""

    def __init__(self, documentos: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenizar(d)) for d in documentos]
        self.largo = [sum(tf.values()) for tf in self.tf]
        self.N = len(documentos)
        self.avgdl = sum(self.largo) / self.N
        self.df = Counter()
        for tf in self.tf:
            self.df.update(tf.keys())

    def idf(self, termino: str) -> float:
        n = self.df.get(termino, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def buscar(self, consulta: str, k: int = 3) -> list[tuple[int, float]]:
        q = tokenizar(consulta)
        resultados = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.largo[i] / self.avgdl))
            if s > 0:
                resultados.append((i, s))
        return sorted(resultados, key=lambda x: -x[1])[:k]

    def terminos_desconocidos(self, consulta: str) -> list[str]:
        """Términos de la pregunta que no aparecen en NINGÚN fragmento (señal útil para el rechazo)."""
        return sorted({t for t in tokenizar(consulta) if self.df.get(t, 0) == 0})
