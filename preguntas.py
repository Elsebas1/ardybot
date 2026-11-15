"""Preguntas de prueba (semilla del conjunto de evaluación de la etapa 6)."""

# (pregunta, [textos que debe contener el fragmento correcto])
EN_DOMINIO = [
    ("¿En qué bloque queda el salón 1-11?", ["Salón 1-11", "Bloque 1"]),
    ("¿En qué bloque queda el salón 2-12?", ["Salón 2-12", "Bloque 2"]),
    ("¿En qué piso queda el CEP-11?", ["Salón CEP-11", "1er piso"]),
    ("¿Dónde queda el salón DER-11?", ["Salón DER-11", "Derecho"]),
    ("¿Dónde queda el consultorio jurídico?", ["Consultorio Jurídico"]),
]

# Preguntas cuya respuesta NO está en el documento (el bot debe declinar).
FUERA = [
    "¿Dónde queda el cine?",
    "¿Dónde queda el salón 10292021?",
    "¿Dónde queda la facultad de Geología?",
    "¿Cómo se prepara una arepa?",  # ejemplo sin relación con la universidad
]
