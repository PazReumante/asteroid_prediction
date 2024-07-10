import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Configurar la página Streamlit
st.set_page_config(page_title="Mi Aplicación", page_icon=":gráfico_de_barras:", layout="wide")

# Cargar el modelo entrenado
model = joblib.load('moid_model.pkl')

# Cargar los datos desde clean_test.csv
@st.cache(persist=True)
def load_data(file_name):
    return pd.read_csv(file_name)

test = load_data('clean_test.csv')

# Función para la página "Database Information"
def pagina_1():
    st.title('Database Information')
    st.markdown(
        """
        El MOID (Minimum Orbit Intersection Distance) es una medida utilizada en astronomía para determinar la menor distancia entre las órbitas de dos cuerpos celestes, típicamente planetas, asteroides o cometas. Este parámetro es crucial para evaluar el riesgo de colisión potencial entre objetos espaciales y es fundamental en la predicción y seguimiento de objetos cercanos a la Tierra.

        ![MOID](https://github.com/PazReumante/asteroid_prediction/blob/main/Project/images/Minimum-Orbital-Intersection-Distance.png)
        """
    )

    if st.button("Mostrar Importancia del MOID"):
        mostrar_importancia()

def mostrar_importancia():
    st.markdown(
        """
        ## Importancia del MOID

        ### a. Evaluación de Riesgo de Impacto
        La MOID es crucial para evaluar el riesgo de impacto de un objeto con la Tierra. Una MOID pequeña sugiere que en algún momento futuro, el objeto podría pasar cerca de la Tierra, lo que podría representar un riesgo de impacto.

        ### b. Planes de Mitigación
        Conocer la MOID ayuda a las agencias espaciales a desarrollar estrategias de mitigación para posibles impactos, como desviar el objeto o prepararse para una colisión.

        ### c. Monitoreo y Seguimiento
        Los astrónomos y las agencias espaciales utilizan la MOID para monitorear y rastrear objetos potencialmente peligrosos (PHA, Potentially Hazardous Asteroids).

        En el contexto de la base de datos MOID, el Análisis Exploratorio de Datos (EDA) se enfocará en las siguientes etapas:

        1. **Exploración de Variables**: Se estudiarían las variables relevantes relacionadas con las órbitas de los cuerpos celestes, como las coordenadas orbitales, las velocidades relativas y el MOID entre diferentes objetos.

        2. **Análisis Estadístico**: Se aplicarían técnicas estadísticas para calcular medidas descriptivas como media, mediana, desviación estándar, así como para identificar tendencias y distribuciones de los datos.

        3. **Visualización de Datos**: Se utilizarían gráficos y diagramas (como scatter plots, histogramas, y gráficos de densidad) para visualizar la distribución de los MOID y otras variables relevantes, facilitando la detección de patrones o agrupaciones.

        4. **Identificación de Outliers**: Se buscarían valores atípicos que podrían indicar eventos de interés astronómico, como aproximaciones extremadamente cercanas entre órbitas.

        5. **Correlaciones y Relaciones**: Se explorarían las relaciones entre las diferentes variables para entender cómo afectan los cambios en una variable a otra, especialmente en términos de riesgo de colisión o trayectorias próximas.

        Posteriormente, evaluaremos varios modelos de machine learning adecuados, ajustaremos sus hiperparámetros mediante validación cruzada, y finalmente interpretaremos y seleccionaremos el modelo más apropiado basándonos en métricas de evaluación establecidas. Este enfoque garantizará que el modelo elegido se adapte eficazmente a las necesidades específicas del proyecto, maximizando su rendimiento y utilidad práctica.

        ### Se trabajará con la siguiente base de datos:
        [Dataset MOID](https://gitlab.com/mirsakhawathossain/pha-ml/-/raw/master/Dataset/dataset.csv)

        En un breve resumen:
        - La base de datos contiene 958.524 entradas con 45 variables, las cuales serán detalladas a continuación.
        - Se dividirá en dataset en 5 partes con el fin de cumplir los parámetros de almacenamiento de github.
        - Para el procesamiento de datos se cogerá una muestra aleatoria de 150.000 entradas con el propósito de optimizar el funcionamiento del código.
        - Se utilizará SQL para la creación de la muestra y como base de datos del desarrollo del proyecto.
        """
    )

# Función para la página "MOID Prediction"
def Moid_prediction():
    st.title('MOID Prediction')
    st.markdown("""
    Esta aplicación permite calcular y predecir el MOID (Minimum Orbit Intersection Distance) de asteroides basándose en sus características clave. A continuación, se presenta una breve descripción de cada una de ellas:
    """)

    # Definir descripciones de las variables
    descriptions = {
        'H': 'Magnitud absoluta, se refiere al brillo, lo que refiere si es objeto es visible o no.',
        'a': 'Semieje mayor es la distancia promedio del sol, en unidades astronómicas.',
        'q': 'Distancia al perihelio refiere a la máxima aproximación al sol, en unidades astronómicas.',
        'ad': 'Distancia al afelio, distancia máxima del sol, en unidades astronómicas.',
        'n': 'Movimiento medio es la tasa promedio de movimiento a lo largo de la órbita, en grados por día',
        'tp_cal': 'Tiempo del paso por el perihelio, en fecha de calendario.',
        'per_y': 'Período orbital',
        'class_n': 'Tipo de asteroide, con ID numérico'
    }

    # Verificar si hay datos de prueba cargados
    if test.empty:
        st.error("No se pudieron cargar los datos de prueba. Por favor, revisa el archivo `clean_test.csv`.")
        return

    # Crear 8 columnas para organizar los botones de manera horizontal
    cols = st.columns(8)

    # Crear los 8 botones dentro de las columnas
    for i, var in enumerate(descriptions):
        if cols[i].button(f"{var}"):
            st.write(f"**{var}**: {descriptions[var]}")

    st.sidebar.header("Seleccione las características para la predicción:")

    # Crear sliders para cada variable seleccionada con su descripción
    input_data = {}
    for var in descriptions:
        input_data[var] = st.sidebar.slider(f"{var} - {descriptions[var]}", float(test[var].min()), float(test[var].max()), float(test[var].mean()))

    # Convertir los sliders a un DataFrame
    input_df = pd.DataFrame([input_data])

    # Realizar la predicción
    if st.sidebar.button("Realizar Predicción"):
        prediction = model.predict(input_df)[0]
        st.subheader("Resultado de la Predicción")
        st.write(f"La predicción del MOID es: {prediction:.6f}")

        # Determinar si el asteroide está cercano a la Tierra
        threshold = 0.05  # Define un umbral para considerar si el asteroide está cercano
        if prediction < threshold:
            st.warning("¡El asteroide está cercano a la Tierra!")
        else:
            st.success("El asteroide no está cercano a la Tierra.")

        # Preparar los datos para el gráfico de puntos
        selected_variables = list(input_data.keys())
        colors = np.linspace(0, 1, len(selected_variables))
        plt.figure(figsize=(8, 6))
        for i, var in enumerate(selected_variables):
            plt.scatter(input_data[var], prediction, color=plt.cm.spring(colors[i]), label=f"{var} - {descriptions[var]}")

        plt.title('Predicción MOID vs Variables Seleccionadas')
        plt.xlabel('Valor de la Variable')
        plt.ylabel('Predicción MOID')
        plt.legend()
        st.pyplot(plt)

    st.sidebar.info("""
    Seleccione los valores de las características del asteroide desde los datos de `test.csv` usando los sliders.
    """)

# Llamar a la función correspondiente según la selección del menú
if st.sidebar.selectbox('Seleccione una página:', ['Database Information', 'MOID Prediction']) == 'Database Information':
    pagina_1()
else:
    Moid_prediction()
