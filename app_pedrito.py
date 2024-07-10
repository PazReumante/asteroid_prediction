import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

# Función para realizar predicciones
def predict_collision(data, model, scaler):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction

# Escalador
scaler = StandardScaler()
scaler.fit(train.drop(columns=['moid']))

# Formulario para la entrada de datos del usuario
def user_input_features():
    H = st.sidebar.number_input("Magnitud Absoluta (H)", min_value=0.0, max_value=100.0, value=16.9, step=0.1)
    a = st.sidebar.number_input("Eje Semi-Mayor (a)", min_value=0.0, max_value=100.0, value=3.066, step=0.001)
    q = st.sidebar.number_input("Distancia del Perihelio (q)", min_value=0.0, max_value=100.0, value=2.390, step=0.001)
    ad = st.sidebar.number_input("Distancia del Afelio (ad)", min_value=0.0, max_value=100.0, value=3.843, step=0.001)
    n = st.sidebar.number_input("Movimiento Medio (n)", min_value=0.0, max_value=10.0, value=0.24, step=0.001)
    tp_cal = st.sidebar.number_input("Tiempo de Paso por el Perihelio (tp_cal)", min_value=0.0, max_value=1e8, value=2.019578e7, step=1.0)
    per_y = st.sidebar.number_input("Período Orbital (per_y)", min_value=0.0, max_value=1e8, value=47.82621, step=0.01)
    class_n = st.sidebar.number_input("Clasificación del Asteroide (class_n)", min_value=0, max_value=12, value=0, step=1)
    
    data = {
        'H': H,
        'a': a,
        'q': q,
        'ad': ad,
        'n': n,
        'tp_cal': tp_cal,
        'per_y': per_y,
        'class_n': class_n
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Mostrar los datos de entrada
st.subheader("Características de Entrada")
st.write(input_df)

# Predicción
if st.button("Predecir Colisión"):
    prediction = predict_collision(input_df, model, scaler)
    st.subheader("Predicción")
    if prediction[0] <= 0.05:  # Asumiendo que un MOID <= 0.05 indica un posible impacto
        st.write("*Advertencia:* ¡El asteroide podría colisionar con la Tierra!")
    else:
        st.write("Es poco probable que el asteroide colisione con la Tierra.")

# Gráficos interactivos
st.subheader("Visualización de Datos")
st.write("""
Para mejorar la comprensión de los datos, a continuación se presentan algunos gráficos interactivos:
""")

# Gráfico de distribución de la Magnitud Absoluta (H)
fig, ax = plt.subplots()
sns.histplot(train['H'], bins=20, kde=True, ax=ax)
ax.set_title('Distribución de la Magnitud Absoluta (H)')
st.pyplot(fig)

# Gráfico de dispersión entre Eje Semi-Mayor (a) y Distancia del Perihelio (q)
fig, ax = plt.subplots()
sns.scatterplot(x='a', y='q', data=train, ax=ax)
ax.set_title('Dispersión entre Eje Semi-Mayor (a) y Distancia del Perihelio (q)')
st.pyplot(fig)

# Mejorar la visualización
st.image(
    """
    <style>
    .reportview-container {
        background: url("https://www.nasa.gov/sites/default/files/thumbnails/image/pia24621.jpg");
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.9);
    }
    </style>
    """,
    unsafe_allow_html=True
)