"""Compara BM25, embeddings e híbrido sobre las mismas preguntas.

Uso:   python3 demo_etapa2.py              (modelo real; requiere sentence-transformers)
       python3 demo_etapa2.py --simulado   (solo prueba el flujo, NO es semántico)
"""
import sys
import time

from embeddings import DenseRetriever, HybridRetriever, encoder_sbert, encoder_simulado
from preguntas import EN_DOMINIO, FUERA
from recuperacion import BM25, cargar_chunks, normalizar

chunks = cargar_chunks("data/mapas_universidad_ibague.md")
textos = [c["texto"] for c in chunks]
bm25 = BM25(textos)

if "--simulado" in sys.argv:
    print("MODO SIMULADO: vectores TF-IDF de caracteres, NO semánticos (solo prueba el flujo).\n")
    enc = encoder_simulado(textos)
else:
    try:
        enc = encoder_sbert()
    except ImportError:
        sys.exit("Falta sentence-transformers. Instálalo con: python3 -m pip install sentence-transformers")

denso = DenseRetriever(textos, enc)
metodos = {"BM25": bm25.buscar, "Denso": denso.buscar, "Híbrido": HybridRetriever(bm25, denso, alpha=0.5).buscar}


def posicion(res, esperado):
    """Posición (1, 2, 3) del primer fragmento que contiene todo lo esperado; None si no aparece."""
    for pos, (i, _) in enumerate(res, 1):
        if all(normalizar(e) in normalizar(chunks[i]["texto"]) for e in esperado):
            return pos
    return None


print(f"{'Pregunta':<40}" + "".join(f"{m:>10}" for m in metodos))
resumen = {m: {"top1": 0, "top3": 0, "ms": 0.0} for m in metodos}
for q, esperado in EN_DOMINIO:
    fila = f"{q:<40}"
    for m, buscar in metodos.items():
        t0 = time.perf_counter()
        res = buscar(q, 3)
        resumen[m]["ms"] += (time.perf_counter() - t0) * 1000
        pos = posicion(res, esperado)
        resumen[m]["top1"] += pos == 1
        resumen[m]["top3"] += pos is not None
        fila += f"{(pos if pos else '-'):>10}"
    print(fila)

n = len(EN_DOMINIO)
print(f"\n{'Método':<10}{'Top-1':>8}{'Top-3':>8}{'ms/consulta':>14}")
for m, r in resumen.items():
    print(f"{m:<10}{str(r['top1']) + '/' + str(n):>8}{str(r['top3']) + '/' + str(n):>8}{r['ms'] / n:>14.2f}")

# Puntajes máximos "crudos": servirán de características para el clasificador de rechazo (etapa 3).
print(f"\n{'Pregunta':<40}{'BM25 máx':>10}{'Coseno máx':>12}   Términos desconocidos")
for q in [p for p, _ in EN_DOMINIO] + FUERA:
    b = bm25.buscar(q, 1)
    print(f"{q:<40}{(b[0][1] if b else 0.0):>10.2f}{denso.puntajes(q).max():>12.2f}   {bm25.terminos_desconocidos(q)}")
