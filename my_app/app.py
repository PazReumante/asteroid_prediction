import streamlit as st
import importlib

st.set_page_config(
    page_title="Orbital Minimum Intersection Distance Analysis MOID",
    page_icon="🌍",
)

st.sidebar.title("Navegación")
st.sidebar.header("Páginas")

# Crear un diccionario con las opciones de las páginas y sus archivos
PAGES = {
    "Database information": "pages.page1",
    "MOID prediction": "pages.page2"  # Puedes agregar más páginas aquí
}

# Selector de páginas en la barra lateral
page = st.sidebar.selectbox("Selecciona una página", list(PAGES.keys()))

# Importar dinámicamente la página seleccionada
page_module = importlib.import_module(PAGES[page])

# Llamar a la función `show` de la página seleccionada
page_module.show()

