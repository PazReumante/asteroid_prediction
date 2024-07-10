import streamlit as st

def show():
    st.title("Orbital Minimum Intersection Distance Analysis MOID - DATA SCIENCE PROJECT")

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

