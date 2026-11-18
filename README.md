# Ardy Bot

Asistente de información de la Universidad de Ibagué (salones, bloques, oficinas y lugares del campus).
Arquitectura RAG: recuperación híbrida (BM25 + embeddings) + clasificador de rechazo + LLM local (Ollama).

## Requisitos
- Python 3.9 o superior (probado con 3.14 en macOS).
- [Ollama](https://ollama.com) abierto, con el modelo `llama3.2` (`ollama pull llama3.2`).
- Conexión a internet la primera vez (descarga el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2`).

## Instalación
```
MAC
git clone https://github.com/Elsebas1/ardybot.git
cd ardybot
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

Windows
git clone https://github.com/Elsebas1/ardybot.git
cd ardybot
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Uso
Con el ambiente activo y Ollama abierto:
```
streamlit run app.py        # interfaz de chat en http://localhost:8501
python3 asistente.py        # chat en la terminal
```
La barra lateral de la interfaz muestra el documento procesado, el modelo y el umbral del clasificador de rechazo (por defecto 0,2).

## Contenido
| Archivo | Para qué sirve |
|---|---|
| `data/mapas_universidad_ibague.md` | Documento procesado: transcripción de los mapas de salones (2022) e institucional (2025), más relaciones de vecindad y un dato de servicio aportados por el estudiante. |
| `data/preguntas_prueba.csv` | 50 preguntas de prueba (30 dentro del documento, 20 fuera) con respuesta esperada y división entrenamiento/prueba. |
| `recuperacion.py` | Lee el documento, lo parte en fragmentos y busca con BM25. |
| `embeddings.py` | Recuperación semántica e híbrida. |
| `rechazo.py` | Clasificador de rechazo (regresión logística). |
| `asistente.py` | Une todo: clasificar, recuperar, generar con Ollama y validar la salida. |
| `app.py` | Interfaz de chat (Streamlit). |

## Reproducir las evaluaciones
```
python3 demo_etapa1.py            # debe mostrar 202 fragmentos y Top-1 5/5
python3 evaluar_recuperacion.py   # BM25 vs embeddings vs híbrido
python3 evaluar_rechazo.py        # mecanismos de rechazo y barrido de umbral
python3 evaluar_bot.py            # bot completo con Ollama; guarda data/resultados_bot.csv
```
La división entrenamiento/prueba usa semilla fija (42) y el generador usa temperatura 0, por lo que los resultados son reproducibles
(pueden variar levemente con otras versiones de las librerías o del modelo).

## Resultados
Evaluación sobre 50 preguntas de prueba (30 dentro del documento, 20 fuera):

| Métrica | Valor |
|---|---|
| Exactitud estricta (30 preguntas del documento) | 83 % (25/30) |
| Exactitud flexible (cuenta respuestas parciales) | 93 % (28/30) |
| Preguntas fuera del documento declinadas | 20 / 20 |

Calificación manual: 25 respuestas correctas, 3 parciales y 2 incorrectas, con el generador
respondiendo todas las preguntas del documento (umbral 0). Con el umbral por defecto (0,2),
el clasificador de rechazo descarta además una pregunta válida.

Recuperación de fragmentos (fragmento correcto entre los 3 primeros, sobre 30 preguntas):

| Método | Top-3 |
|---|---|
| BM25 | 26 / 30 |
| Embeddings | 21 / 30 |
| Híbrido | 28 / 30 |

Los errores y respuestas parciales son del mismo tipo: confundir un lugar con su vecino.

## Limitaciones
- Las 50 preguntas las escribieron los autores conociendo el documento, lo que favorece al bot; son pocas para conclusiones fuertes.
- El clasificador de rechazo es débil por sí solo; en las pruebas, la barrera principal fue el prompt del generador.
- Un modelo pequeño (`llama3.2`) puede confundir un lugar con su vecino cuando los fragmentos mencionan "al lado de X (punto N)".
- El documento no trae horarios, precios ni personas: el bot debe declinar esas preguntas.

## Autor
Juan Sebastián Gómez y Juan David Romero Universidad de Ibagué.
