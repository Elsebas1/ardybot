"""Etapa 2b: evalúa BM25, embeddings e híbrido con data/preguntas_prueba.csv (solo preguntas 'dentro').

Uso:   python3 evaluar_recuperacion.py              (modelo real)
       python3 evaluar_recuperacion.py --simulado   (solo prueba el flujo; los números de Denso/Híbrido no valen)
"""
import csv
import sys
import time

from embeddings import DenseRetriever, HybridRetriever, encoder_sbert, encoder_simulado
from recuperacion import BM25, cargar_chunks, normalizar

chunks = cargar_chunks("data/mapas_universidad_ibague.md")
textos = [c["texto"] for c in chunks]
bm25 = BM25(textos)

if "--simulado" in sys.argv:
    print("MODO SIMULADO: Denso e Híbrido NO son semánticos; solo BM25 es real.\n")
    enc = encoder_simulado(textos)
else:
    try:
        enc = encoder_sbert()
    except ImportError:
        sys.exit("Falta sentence-transformers (activa el ambiente virtual: source .venv/bin/activate)")
denso = DenseRetriever(textos, enc)

with open("data/preguntas_prueba.csv", encoding="utf-8-sig", newline="") as fh:
    filas = [f for f in csv.DictReader(fh) if f["tipo"] == "dentro"]


def esperado_de(f):
    return [e.strip() for e in f["fragmento_esperado"].split("|")]


def posicion(res, esperado):
    for pos, (i, _) in enumerate(res, 1):
        if all(normalizar(e) in normalizar(chunks[i]["texto"]) for e in esperado):
            return pos
    return None


# Aviso si, tras editar el CSV, alguna pregunta quedó sin fragmento correcto en el documento.
for f in filas:
    if not any(all(normalizar(e) in normalizar(c["texto"]) for e in esperado_de(f)) for c in chunks):
        print(f"AVISO {f['id']}: ningún fragmento contiene {esperado_de(f)}. Revisa 'fragmento_esperado'.")


def evaluar(buscar, grupo, k=10):
    t1 = t3 = 0
    rr = 0.0
    fallos = []
    for f in grupo:
        pos = posicion(buscar(f["pregunta"], k), esperado_de(f))
        t1 += pos == 1
        t3 += pos is not None and pos <= 3
        rr += 1 / pos if pos else 0.0
        if pos is None or pos > 3:
            fallos.append(f["id"])
    n = len(grupo)
    return {"n": n, "t1": t1, "t3": t3, "mrr": rr / n if n else 0.0, "fallos": fallos}


metodos = {"BM25": bm25.buscar, "Denso": denso.buscar, "Híbrido": HybridRetriever(bm25, denso, 0.5).buscar}
lit = [f for f in filas if f["estilo"] == "literal"]
par = [f for f in filas if f["estilo"] == "parafraseada"]

print(f"{'':<10}{'--- literales (' + str(len(lit)) + ') ---':<26}{'--- parafraseadas (' + str(len(par)) + ') ---':<28}")
print(f"{'Método':<10}{'Top-1':>6}{'Top-3':>7}{'MRR':>7}   {'Top-1':>6}{'Top-3':>7}{'MRR':>7}   {'ms/consulta':>12}")
fallos = {}
for m, buscar in metodos.items():
    for f in filas:  # pasada de calentamiento (no se mide): las primeras llamadas al modelo son lentas
        buscar(f["pregunta"], 3)
    t0 = time.perf_counter()
    for f in filas:
        buscar(f["pregunta"], 3)
    ms = (time.perf_counter() - t0) * 1000 / len(filas)
    a, b = evaluar(buscar, lit), evaluar(buscar, par)
    fallos[m] = a["fallos"] + b["fallos"]
    print(f"{m:<10}{a['t1']:>6}{a['t3']:>7}{a['mrr']:>7.2f}   {b['t1']:>6}{b['t3']:>7}{b['mrr']:>7.2f}   {ms:>12.2f}")
print("\nPreguntas fuera del top-3:")
for m, ids in fallos.items():
    print(f"  {m:<8}{ids}")

# Sensibilidad al peso alpha (1 = solo BM25, 0 = solo embeddings). Se elige con ENTRENAMIENTO y se reporta PRUEBA.
ent = [f for f in filas if f["division"] == "entrenamiento"]
pru = [f for f in filas if f["division"] == "prueba"]
print(f"\n{'alpha':<8}{'Entren. Top-1':>15}{'MRR':>7}   {'Prueba Top-1':>14}{'MRR':>7}")
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    h = HybridRetriever(bm25, denso, a).buscar
    r1, r2 = evaluar(h, ent), evaluar(h, pru)
    print(f"{a:<8}{str(r1['t1']) + '/' + str(r1['n']):>15}{r1['mrr']:>7.2f}   {str(r2['t1']) + '/' + str(r2['n']):>14}{r2['mrr']:>7.2f}")
