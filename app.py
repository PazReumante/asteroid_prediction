import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configurar la página Streamlit (debe ser la primera llamada de Streamlit en tu script)
st.set_page_config(page_title="Calculadora y Predicción de MOID", layout="centered")

# Cargar el modelo entrenado
model = joblib.load('moid_model.pkl')

# Cargar los datos desde train.csv y test.csv
@st.cache_data
def load_data(file_name):
    return pd.read_csv(file_name)

train = load_data('clean_train.csv')
test = load_data('clean_test.csv')

st.title("Calculadora y Predicción de MOID")

st.sidebar.header("Seleccione las características para la predicción de test.csv")

# Definir descripciones de las variables
descriptions = {
    'H': 'Magnitud absoluta',
    'a': 'Semieje mayor (UA)',
    'q': 'Perihelio (UA)',
    'ad': 'Afelio (UA)',
    'n': 'Media anomalía diaria (grados/día)',
    'tp_cal': 'Tiempo de paso cercano calculado (JD)',
    'per_y': 'Período orbital (años)',
    'class_n': 'Tipo de asteroide (numérico)'
}

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
Seleccione los valores de las características del asteroide desde los datos de `test.csv` usando los sliders y haga clic en "Realizar Predicción" para calcular el MOID y determinar si el asteroide está cercano a la Tierra.
""")
