# Orbital Minimum Intersection Distance Analysis MOID - DATA SCIENCE PROJECT

El MOID (Minimum Orbit Intersection Distance) es una medida utilizada en astronomía para determinar la menor distancia entre las órbitas de dos cuerpos celestes, típicamente planetas, asteroides o cometas. Este parámetro es crucial para evaluar el riesgo de colisión potencial entre objetos espaciales y es fundamental en la predicción y seguimiento de objetos cercanos a la Tierra.

![MOID](https://www.researchgate.net/figure/Conjunction-always-occurs-in-the-vicinity-of-points-of-closest-approach-between-orbits_fig3_311395113)

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

En un breve resumen:

• La base de datos contiene 958.524 entradas con 45 variables, las cuales serán detalladas a continuación.

• Se dividirá en dataset en 5 partes con el fin de cumplir los parametros de almanecemiento de github.

• Para el procesamiento de datos se cogerá una muestra aleatorio de 150.000 entradas con el propósito de optimizar el funcionamiento del código.

• Se utilizará SQL para la creación de la muestra y como base de datos del desarrollo del proyecto.

### Descripción de Variables

- **_id_**: Identificador único del objeto.
- **_spkid_**: SPK-ID, un identificador único utilizado por el JPL de la NASA (Jet Propulsion Laboratory).
- **_full\_name_**: Nombre completo o designación del objeto.
- **_pdes_**: Designación principal del objeto.
- **_name_**: Nombre común del objeto (si tiene).
- **_prefix_**: Prefijo del objeto (si tiene).
- **_neo_**: Indicador de si el objeto es un objeto cercano a la Tierra (NEO).
- **_pha_**: Indicador de si el objeto es un asteroide potencialmente peligroso (PHA).
- **_H_**: Magnitud absoluta (brillo) del objeto.
- **_diameter_**: Diámetro estimado del objeto (en kilómetros).
- **_albedo_**: Albedo (reflectividad) del objeto.
- **_diameter\_sigma_**: Incertidumbre en la estimación del diámetro.
- **_orbit\_id_**: Identificador de los datos orbitales utilizados.
- **_epoch_**: Época de los elementos orbitales (en Fecha Juliana).
- **_epoch\_mjd_**: Época de los elementos orbitales (en Fecha Juliana Modificada).
- **_epoch\_cal_**: Época de los elementos orbitales (en fecha de calendario).
- **_equinox_**: Equinoccio de los elementos orbitales.
- **_e_**: Excentricidad orbital.
- **_a_**: Semieje mayor (distancia promedio del sol, en unidades astronómicas).
- **_q_**: Distancia al perihelio (máxima aproximación al sol, en unidades astronómicas).
- **_i_**: Inclinación orbital (inclinación de la órbita en relación con la eclíptica, en grados).
- **_om_**: Longitud del nodo ascendente (en grados).
- **_w_**: Argumento del perihelio (en grados).
- **_ma_**: Anomalía media (en grados).
- **_ad_**: Distancia al afelio (distancia máxima del sol, en unidades astronómicas).
- **_n_**: Movimiento medio (tasa promedio de movimiento a lo largo de la órbita, en grados por día).
- **_tp_**: Tiempo del paso por el perihelio (en Fecha Juliana).
- **_tp\_cal_**: Tiempo del paso por el perihelio (en fecha de calendario).
- **_per_**: Período orbital (tiempo para completar una órbita, en días).
- **_per\_y_**: Período orbital (tiempo para completar una órbita, en años).
- **_moid_**: Distancia mínima de intersección de la órbita (distancia más cercana a la órbita de la Tierra, en unidades astronómicas).
- **_moid\_ld_**: Distancia mínima de intersección de la órbita (en distancias lunares).
- **_sigma\_e_**: Incertidumbre en la excentricidad orbital.
- **_sigma\_a_**: Incertidumbre en el semieje mayor.
- **_sigma\_q_**: Incertidumbre en la distancia al perihelio.
- **_sigma\_i_**: Incertidumbre en la inclinación orbital.
- **_sigma\_om_**: Incertidumbre en la longitud del nodo ascendente.
- **_sigma\_w_**: Incertidumbre en el argumento del perihelio.
- **_sigma\_ma_**: Incertidumbre en la anomalía media.
- **_sigma\_ad_**: Incertidumbre en la distancia al afelio.
- **_sigma\_n_**: Incertidumbre en el movimiento medio.
- **_sigma\_tp_**: Incertidumbre en el tiempo del paso por el perihelio.
- **_sigma\_per_**: Incertidumbre en el período orbital.
- **_class_**: Clasificación dinámica del objeto (por ejemplo, "AMO" para asteroides Amor, "APO" para asteroides Apolo, etc.).
- **_rms_**: Raíz cuadrática media de los residuos (medida del ajuste de la órbita).

Después de realizar el EDA, se determino que se trabajará el entrenamiento de modelos con las siguientes variables, mostradas en este caso por su matriz de correlación:

![](https://github.com/PazReumante/asteroid_prediction/blob/main/matriz%20de%20correlacion%20-%20asteoride.png)



 
