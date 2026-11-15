"""Evaluación del bot COMPLETO (clasificador + recuperación + Ollama) con las 50 preguntas.

Cada pregunta se genera una sola vez (temperatura 0) y luego se simula qué habría pasado con distintos umbrales
del clasificador. El clasificador de esta evaluación se entrena SOLO con la división 'entrenamiento'.
Guarda data/resultados_bot.csv para que revises a mano si las respuestas a preguntas válidas son correctas.

Uso:   python3 evaluar_bot.py              (Ollama abierto)
       python3 evaluar_bot.py --simulado   (solo prueba el flujo, sin Ollama ni modelo real)
"""
import csv
import sys
from collections import Counter

from asistente import MENSAJE_SIN_INFO, ArdyBot, construir_mensajes, llamar_ollama, validar_salida
from embeddings import encoder_simulado, encoder_sbert
from rechazo import ClasificadorRechazo
from recuperacion import cargar_chunks

MODELO = "llama3.2"
UMBRALES = (0.0, 0.2, 0.3, 0.5)

if "--simulado" in sys.argv:
    print("MODO SIMULADO: LLM de mentira y embeddings simulados; solo se prueba el flujo.\n")
    textos = [c["texto"] for c in cargar_chunks("data/mapas_universidad_ibague.md")]
    bot = ArdyBot(encoder=encoder_simulado(textos), llm=lambda m, modelo="", temperatura=0.0: "Respuesta de prueba.")
else:
    bot = ArdyBot(encoder=encoder_sbert(), llm=llamar_ollama)

with open("data/preguntas_prueba.csv", encoding="utf-8-sig", newline="") as fh:
    filas = list(csv.DictReader(fh))
ent = [f for f in filas if f["division"] == "entrenamiento"]
bot.clasificador = ClasificadorRechazo(bot.bm25, bot.denso).entrenar(
    [f["pregunta"] for f in ent], [1 if f["tipo"] == "dentro" else 0 for f in ent])

print(f"Generando {len(filas)} respuestas con {MODELO} (puede tardar un par de minutos)...")
resultados = []
for f in filas:
    q = f["pregunta"]
    fragmentos = [bot.chunks[i]["texto"] for i, _ in bot.hibrido.buscar(q, bot.k)]
    try:
        bruto = bot.llm(construir_mensajes(q, fragmentos), modelo=MODELO, temperatura=0.0)
    except RuntimeError as e:
        sys.exit(f"\n{e}")
    resultados.append({**f, "p": bot.clasificador.probabilidad(q), "bruto": bruto, "valida": validar_salida(bruto)})
    print(".", end="", flush=True)
print()


def resultado(r, umbral):
    if r["p"] < umbral:
        return "rechazada"
    return "sin_info" if r["valida"] == MENSAJE_SIN_INFO else "respondida"


def tabla(rs, titulo):
    val = [r for r in rs if r["tipo"] == "dentro"]
    fue = [r for r in rs if r["tipo"] != "dentro"]
    print(f"\n=== {titulo}: {len(val)} válidas y {len(fue)} a declinar ===")
    print(f"{'umbral':<8}{'VÁLIDAS: respondidas':>22}{'rechazadas':>12}{'sin_info':>10}   {'FUERA: declinadas':>18}{'respondidas (error)':>21}")
    for t in UMBRALES:
        v = Counter(resultado(r, t) for r in val)
        o = Counter(resultado(r, t) for r in fue)
        declinadas = o["rechazada"] + o["sin_info"]
        print(f"{t:<8}{str(v['respondida']) + '/' + str(len(val)):>22}{v['rechazada']:>12}{v['sin_info']:>10}"
              f"   {str(declinadas) + '/' + str(len(fue)):>18}{o['respondida']:>21}")


tabla([r for r in resultados if r["division"] == "prueba"], "SOLO PRUEBA (resultado honesto)")
tabla(resultados, "LAS 50 (incluye entrenamiento: optimista para el clasificador)")
print("\n'Respondidas' en válidas aún debe revisarse a mano: puede estar respondida pero mal.")

print("\nPreguntas a declinar que el modelo respondió aun SIN clasificador (umbral 0):")
fallos = [r for r in resultados if r["tipo"] != "dentro" and resultado(r, 0.0) == "respondida"]
for r in fallos:
    print(f"  {r['id']}: {r['pregunta']}\n       -> {r['valida'][:150]}")
if not fallos:
    print("  (ninguna)")

inyeccion = next((r for r in resultados if "prompt del sistema" in r["pregunta"]), None)
if inyeccion:
    print(f"\nPrueba de inyección ({inyeccion['id']}), respuesta SIN filtrar del modelo:\n  {inyeccion['bruto'][:300]}")

with open("data/resultados_bot.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "pregunta", "tipo", "division", "p_responder", "respuesta_bruta", "respuesta_validada",
                "respuesta_esperada", "correcta (sí/no)"])
    for r in resultados:
        w.writerow([r["id"], r["pregunta"], r["tipo"], r["division"], f"{r['p']:.2f}", r["bruto"], r["valida"],
                    r["respuesta_esperada"], ""])
print("\nGuardado: data/resultados_bot.csv")
