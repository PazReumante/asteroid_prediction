# Orbital Minimum Intersection Distance Analysis MOID - DATA SCIENCE PROJECT

El MOID (Minimum Orbit Intersection Distance) es una medida utilizada en astronomía para determinar la menor distancia entre las órbitas de dos cuerpos celestes, típicamente planetas, asteroides o cometas. Este parámetro es crucial para evaluar el riesgo de colisión potencial entre objetos espaciales y es fundamental en la predicción y seguimiento de objetos cercanos a la Tierra.

Importancia del MOID

a-.**Evaluación de Riesgo de Impacto**: La MOID es crucial para evaluar el riesgo de impacto de un objeto con la Tierra. Una MOID pequeña sugiere que en algún momento futuro, el objeto podría pasar cerca de la Tierra, lo que podría representar un riesgo de impacto.
b-.**Planes de Mitigación**: Conocer la MOID ayuda a las agencias espaciales a desarrollar estrategias de mitigación para posibles impactos, como desviar el objeto o prepararse para una colisión.
c-.**Monitoreo y Seguimiento**: Los astrónomos y las agencias espaciales utilizan la MOID para monitorear y rastrear objetos potencialmente peligrosos (PHA, Potentially Hazardous Asteroids).

En el contexto de la base de datos MOID, el Análisis Exploratorio de Datos (EDA) se enfocará en las siguientes etapas:

1-.**Exploración de Variables**: Se estudiarían las variables relevantes relacionadas con las órbitas de los cuerpos celestes, como las coordenadas orbitales, las velocidades relativas y el MOID entre diferentes objetos.

2-.**Análisis Estadístico**: Se aplicarían técnicas estadísticas para calcular medidas descriptivas como media, mediana, desviación estándar, así como para identificar tendencias y distribuciones de los datos.

3-.**Visualización de Datos**: Se utilizarían gráficos y diagramas (como scatter plots, histogramas, y gráficos de densidad) para visualizar la distribución de los MOID y otras variables relevantes, facilitando la detección de patrones o agrupaciones.

4-.**Identificación de Outliers**: Se buscarían valores atípicos que podrían indicar eventos de interés astronómico, como aproximaciones extremadamente cercanas entre órbitas.

5-.**Correlaciones y Relaciones**: Se explorarían las relaciones entre las diferentes variables para entender cómo afectan los cambios en una variable a otra, especialmente en términos de riesgo de colisión o trayectorias próximas.

Posteriormente, evaluaremos varios modelos de machine learning adecuados, ajustaremos sus hiperparámetros mediante validación cruzada, y finalmente interpretaremos y seleccionaremos el modelo más apropiado basándonos en métricas de evaluación establecidas. Este enfoque garantizará que el modelo elegido se adapte eficazmente a las necesidades específicas del proyecto, maximizando su rendimiento y utilidad práctica.

Se trabajará con la siguiente base de datos:
https://gitlab.com/mirsakhawathossain/pha-ml/-/raw/master/Dataset/dataset.csv

En un breve resumen;

```python
import pandas as pd

# Carga de datos con el fin de mostrar 
total_data = pd.read_csv("https://gitlab.com/mirsakhawathossain/pha-ml/-/raw/master/Dataset/dataset.csv")

# Mostrar información sobre el conjunto de datos
print(total_data.info())

En función de optimizar el procesamiento de datos se trabajará con una muestra de 150.000 entradas con las siguientes variables que se definen a continuación:

### Descripción de Variables

- ***id***: Identificador único del objeto.
- ***spkid***: SPK-ID, un identificador único utilizado por el JPL de la NASA (Jet Propulsion Laboratory).
- ***full_name***: Nombre completo o designación del objeto.
- ***pdes***: Designación principal del objeto.
- ***name***: Nombre común del objeto (si tiene).
- ***prefix***: Prefijo del objeto (si tiene).
- ***neo***: Indicador de si el objeto es un objeto cercano a la Tierra (NEO).
- ***pha***: Indicador de si el objeto es un asteroide potencialmente peligroso (PHA).
- ***H***: Magnitud absoluta (brillo) del objeto.
- ***diameter***: Diámetro estimado del objeto (en kilómetros).
- ***albedo***: Albedo (reflectividad) del objeto.
- ***diameter_sigma***: Incertidumbre en la estimación del diámetro.
- ***orbit_id***: Identificador de los datos orbitales utilizados.
- ***epoch***: Época de los elementos orbitales (en Fecha Juliana).
- ***epoch_mjd***: Época de los elementos orbitales (en Fecha Juliana Modificada).
- ***epoch_cal***: Época de los elementos orbitales (en fecha de calendario).
- ***equinox***: Equinoccio de los elementos orbitales.
- ***e***: Excentricidad orbital.
- ***a***: Semieje mayor (distancia promedio del sol, en unidades astronómicas).
- ***q***: Distancia al perihelio (máxima aproximación al sol, en unidades astronómicas).
- ***i***: Inclinación orbital (inclinación de la órbita en relación con la eclíptica, en grados).
- ***om***: Longitud del nodo ascendente (en grados).
- ***w***: Argumento del perihelio (en grados).
- ***ma***: Anomalía media (en grados).
- ***ad***: Distancia al afelio (distancia máxima del sol, en unidades astronómicas).
- ***n***: Movimiento medio (tasa promedio de movimiento a lo largo de la órbita, en grados por día).
- ***tp***: Tiempo del paso por el perihelio (en Fecha Juliana).
- ***tp_cal***: Tiempo del paso por el perihelio (en fecha de calendario).
- ***per***: Período orbital (tiempo para completar una órbita, en días).
- ***per_y***: Período orbital (tiempo para completar una órbita, en años).
- ***moid***: Distancia mínima de intersección de la órbita (distancia más cercana a la órbita de la Tierra, en unidades astronómicas).
- ***moid_ld***: Distancia mínima de intersección de la órbita (en distancias lunares).
- ***sigma_e***: Incertidumbre en la excentricidad orbital.
- ***sigma_a***: Incertidumbre en el semieje mayor.
- ***sigma_q***: Incertidumbre en la distancia al perihelio.
- ***sigma_i***: Incertidumbre en la inclinación orbital.
- ***sigma_om***: Incertidumbre en la longitud del nodo ascendente.
- ***sigma_w***: Incertidumbre en el argumento del perihelio.
- ***sigma_ma***: Incertidumbre en la anomalía media.
- ***sigma_ad***: Incertidumbre en la distancia al afelio.
- ***sigma_n***: Incertidumbre en el movimiento medio.
- ***sigma_tp***: Incertidumbre en el tiempo del paso por el perihelio.
- ***sigma_per***: Incertidumbre en el período orbital.
- ***class***: Clasificación dinámica del objeto (por ejemplo, "AMO" para asteroides Amor, "APO" para asteroides Apolo, etc.).
- ***rms***: Raíz cuadrática media de los residuos (medida del ajuste de la órbita).


 