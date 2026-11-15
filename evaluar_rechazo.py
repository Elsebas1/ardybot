"""Etapa 3: compara mecanismos de rechazo (umbrales simples vs regresión logística).
Entrena con la división 'entrenamiento' del CSV y mide en 'prueba'.

Uso:   python3 evaluar_rechazo.py              (modelo real)
       python3 evaluar_rechazo.py --simulado   (solo prueba el flujo; los números NO valen)
"""
import csv
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from embeddings import DenseRetriever, encoder_sbert, encoder_simulado
from recuperacion import BM25, cargar_chunks
from rechazo import NOMBRES, caracteristicas, nuevo_modelo

chunks = cargar_chunks("data/mapas_universidad_ibague.md")
textos = [c["texto"] for c in chunks]
bm25 = BM25(textos)
if "--simulado" in sys.argv:
    print("MODO SIMULADO: los números de coseno NO son semánticos; solo se prueba el flujo.\n")
    enc = encoder_simulado(textos)
else:
    try:
        enc = encoder_sbert()
    except ImportError:
        sys.exit("Falta sentence-transformers (activa el ambiente: source .venv/bin/activate)")
denso = DenseRetriever(textos, enc)

with open("data/preguntas_prueba.csv", encoding="utf-8-sig", newline="") as fh:
    filas = list(csv.DictReader(fh))
y = np.array([1 if f["tipo"] == "dentro" else 0 for f in filas])  # 1 = se puede responder, 0 = declinar
X = np.array([caracteristicas(f["pregunta"], bm25, denso) for f in filas])
ent = np.array([f["division"] == "entrenamiento" for f in filas])
pru = ~ent
print(f"Preguntas: {len(filas)} (entrenamiento {ent.sum()}, prueba {pru.sum()}) | válidas {y.sum()}, a declinar {(y == 0).sum()}\n")

# Promedio de cada señal: ¿separa las preguntas válidas de las que hay que declinar?
print(f"{'Señal':<20}{'válidas':>10}{'a declinar':>12}")
for j, nombre in enumerate(NOMBRES):
    print(f"{nombre:<20}{X[y == 1, j].mean():>10.2f}{X[y == 0, j].mean():>12.2f}")


def metricas(y_real, pred):
    y_real, pred = np.asarray(y_real), np.asarray(pred).astype(int)
    return {
        "ok": int((y_real == pred).sum()), "n": len(y_real),
        "rech": int(((y_real == 1) & (pred == 0)).sum()), "n_val": int((y_real == 1).sum()),
        "acep": int(((y_real == 0) & (pred == 1)).sum()), "n_fuera": int((y_real == 0).sum()),
    }


def fila(nombre, m):
    return (f"{nombre:<34}{str(m['ok']) + '/' + str(m['n']):>9}"
            f"{str(m['rech']) + '/' + str(m['n_val']):>22}{str(m['acep']) + '/' + str(m['n_fuera']):>24}")


def mejor_umbral(v, y_real):
    """Umbral que más aciertos da en entrenamiento (punto medio entre valores consecutivos)."""
    vals = sorted(set(v))
    cortes = [(a + b) / 2 for a, b in zip(vals, vals[1:])] or [vals[0]]
    return max(cortes, key=lambda t: int(((v >= t).astype(int) == y_real).sum()))


# --- Mecanismos que se comparan (todo se ajusta con ENTRENAMIENTO y se mide en PRUEBA) ---
t_bm25 = mejor_umbral(X[ent, 0], y[ent])
t_cos = mejor_umbral(X[ent, 1], y[ent])
modelo = nuevo_modelo().fit(X[ent], y[ent])
proba = modelo.predict_proba(X)[:, 1]
mecanismos = {
    f"Umbral BM25 (>= {t_bm25:.2f})": (X[:, 0] >= t_bm25).astype(int),
    f"Umbral coseno (>= {t_cos:.2f})": (X[:, 1] >= t_cos).astype(int),
    "Sin términos desconocidos": (X[:, 2] == 0).astype(int),
    "Regresión logística (5 señales)": (proba >= 0.5).astype(int),
}
print(f"\n=== EN LAS PREGUNTAS DE PRUEBA ===\n{'Mecanismo':<34}{'Aciertos':>9}{'Válidas rechazadas':>22}{'Fuera aceptadas':>24}")
for nombre, pred in mecanismos.items():
    print(fila(nombre, metricas(y[pru], pred[pru])))

# Validación cruzada de la regresión logística con las 50 preguntas (estimación más estable).
pred_cv = cross_val_predict(nuevo_modelo(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print(f"\n=== VALIDACIÓN CRUZADA (5 pliegues, las {len(y)} preguntas) ===")
print(fila("Regresión logística (5 señales)", metricas(y, pred_cv)))

# Interpretabilidad: coeficientes sobre señales estandarizadas (positivo = empuja a responder).
print("\nCoeficientes de la regresión logística (entrenada con entrenamiento):")
for nombre, c in zip(NOMBRES, modelo[-1].coef_[0]):
    print(f"  {nombre:<20}{c:>7.2f}")

# Análisis de errores de la regresión logística.
print("\nErrores de la regresión logística (p = probabilidad de poder responder):")
hubo = False
for f, p, real in zip(filas, proba, y):
    if int(p >= 0.5) != real:
        hubo = True
        print(f"  {f['id']} [{f['division']}] p={p:.2f}  {'rechazada, era válida' if real else 'aceptada, debía declinar'}: {f['pregunta']}")
if not hubo:
    print("  (ninguno)")

# Barrido del umbral de decisión: cada celda es "válidas rechazadas / fuera aceptadas".
print("\nBarrido del umbral (válidas rechazadas, fuera aceptadas):")
print(f"{'umbral':<8}{'Entrenamiento':>24}{'Prueba':>20}")
for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    pred = (proba >= t).astype(int)
    a, b = metricas(y[ent], pred[ent]), metricas(y[pru], pred[pru])
    ca = f"{a['rech']}/{a['n_val']}  y  {a['acep']}/{a['n_fuera']}"
    cb = f"{b['rech']}/{b['n_val']}  y  {b['acep']}/{b['n_fuera']}"
    print(f"{t:<8}{ca:>24}{cb:>20}")
