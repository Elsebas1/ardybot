"""Interfaz de chat de Ardy Bot.

Ejecutar (con el ambiente virtual activo y Ollama abierto):   streamlit run app.py
"""
import streamlit as st

from asistente import ArdyBot
from embeddings import encoder_sbert

RUTA_DOC = "data/mapas_universidad_ibague.md"

st.set_page_config(page_title="Ardy Bot", page_icon="🎓")


@st.cache_resource(show_spinner="Cargando el documento y el modelo de embeddings...")
def cargar_bot():
    return ArdyBot(encoder=encoder_sbert())


bot = cargar_bot()
if "historial" not in st.session_state:
    st.session_state.historial = []

st.title("Ardy Bot")
st.caption("Asistente de información de la Universidad de Ibagué: salones, bloques, oficinas y lugares del campus.")

with st.sidebar:
    st.header("Documento procesado")
    st.write(f"mapas_universidad_ibague.md: {len(bot.chunks)} fragmentos")
    with st.expander("Ver el documento"):
        with open(RUTA_DOC, encoding="utf-8") as f:
            st.text(f.read())
    st.header("Ajustes")
    modelo = st.text_input("Modelo de Ollama", value="llama3.2")
    umbral = st.slider("Umbral de rechazo", 0.0, 1.0, 0.2, 0.05,
                       help="Más bajo = el clasificador rechaza menos preguntas y deja que el modelo decida. Por defecto 0,2.")
    mostrar = st.checkbox("Mostrar fragmentos usados", value=True)
    if st.button("Borrar conversación"):
        st.session_state.historial = []
        st.rerun()


def pintar_fuentes(fuentes):
    if mostrar and fuentes:
        with st.expander("Fragmentos usados"):
            for texto, puntaje in fuentes:
                st.write(f"- ({puntaje:.2f}) {texto}")


for m in st.session_state.historial:
    with st.chat_message(m["rol"]):
        st.markdown(m["texto"])
        if m["rol"] == "assistant":
            st.caption(m["detalle"])
            pintar_fuentes(m["fuentes"])

pregunta = st.chat_input("Escribe tu pregunta...")
if pregunta:
    st.session_state.historial.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                r = bot.responder(pregunta, umbral=umbral, modelo=modelo)
                texto, fuentes = r["respuesta"], r["fuentes"]
                detalle = f"Estado: {r['estado']} | p(responder) = {r['p']:.2f} | {r['ms']:.0f} ms"
            except RuntimeError as e:
                texto, fuentes, detalle = str(e), [], "Error de conexión con Ollama"
        st.markdown(texto)
        st.caption(detalle)
        pintar_fuentes(fuentes)
    st.session_state.historial.append({"rol": "assistant", "texto": texto, "fuentes": fuentes, "detalle": detalle})
