"""Prueba de la etapa 1: carga el documento, parte en chunks y consulta con BM25."""
from recuperacion import BM25, cargar_chunks, normalizar

chunks = cargar_chunks("data/mapas_universidad_ibague.md")
bm25 = BM25([c["texto"] for c in chunks])
print(f"Chunks creados: {len(chunks)}\n")

# (pregunta, [fragmentos de texto que debe contener el chunk correcto])
en_dominio = [
    ("¿En qué bloque queda el salón 1-11?", ["Salón 1-11", "Bloque 1"]),
    ("¿En qué bloque queda el salón 2-12?", ["Salón 2-12", "Bloque 2"]),
    ("¿En qué piso queda el CEP-11?", ["Salón CEP-11", "1er piso"]),
    ("¿Dónde queda el salón DER-11?", ["Salón DER-11", "Derecho"]),
    ("¿Dónde queda el consultorio jurídico?", ["Consultorio Jurídico"]),
]
fuera = [
    "¿Dónde queda el cine?",
    "¿Dónde queda el salón 10292021?",
    "¿Dónde queda la facultad de Geología?",
    "¿Cómo se prepara una arepa?",  # ejemplo mío: sin relación con la universidad
]

print("=== PREGUNTAS DENTRO DEL DOCUMENTO ===")
aciertos1 = aciertos3 = 0
for q, esperado in en_dominio:
    res = bm25.buscar(q, k=3)
    ok = [all(normalizar(e) in normalizar(chunks[i]["texto"]) for e in esperado) for i, _ in res]
    pos = ok.index(True) + 1 if True in ok else None
    aciertos1 += pos == 1
    aciertos3 += pos is not None
    print(f"\n{q}  ->  chunk correcto en posición: {pos}")
    for i, s in res:
        print(f"   {s:5.2f}  {chunks[i]['texto'][:95]}")
print(f"\nTop-1: {aciertos1}/{len(en_dominio)}   Top-3: {aciertos3}/{len(en_dominio)}")

print("\n=== PREGUNTAS QUE NO ESTÁN EN EL DOCUMENTO ===")
for q in fuera:
    res = bm25.buscar(q, k=1)
    top = f"{res[0][1]:.2f}  {chunks[res[0][0]]['texto'][:60]}" if res else "sin coincidencias"
    print(f"\n{q}\n   mejor resultado: {top}\n   términos desconocidos: {bm25.terminos_desconocidos(q)}")
