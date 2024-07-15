import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Colisión de Asteroides",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Estilo para el fondo del espacio exterior y componentes

st.markdown(
    """
    <style>
    /* Fondo del espacio exterior */
    .reportview-container {
        background: radial-gradient(circle, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0.5) 45%, rgba(0,0,0,0.2) 100%);
        background-size: 100% 300px;
        background-position: 0% 100%;
        color: #ffffff;
    }

    /* Estrellas y galaxias */
    .reportview-container:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0.1) 100%);
        background-size: 100% 100%;
        opacity: 0.5;
    }

    /* Nebulosa */
    .reportview-container:after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to bottom, rgba(128,128,255,0.2) 0%, rgba(128,128,255,0.5) 50%, rgba(128,128,255,0.2) 100%);
        background-size: 100% 100%;
        opacity: 0.3;
    }

    /* Resto de estilos */
    .sidebar.sidebar-content {
        background: linear-gradient(to bottom, rgba(0,0,0,0.8), rgba(0,0,0,0.5));
        padding: 20px;
        border-radius: 10px;
        color: #ffffff;
    }

    .stButton > button {
        background-color: #ffcc00;
        color: #333333;
        border-radius: 5px;
        font-weight: bold;
        padding: 10px 20px;
        transition: background-color 0.3s;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.2);
    }

    .stButton > button:hover {
        background-color: #e6b800;
        color: #ffffff;
        cursor: pointer;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.4);
    }

    .stSelectbox,.stTextInput {
        background-color: #000000;
        color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        border: none;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.2);
    }

    .stMarkdown {
        background-color: rgba(0,0,0,0.5);
        color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
    }

    .sidebar.sidebar-content.stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Crear un menú de navegación en la barra lateral
menu = st.sidebar.selectbox('Seleccione una página:', ['¿Sabías qué...?', 'Estadísticas de los Datos', 'Calculadora de Predicción de Colisión de Asteroide'])

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return joblib.load('moid_model.pkl')

model = load_model()

# Cargar los datos desde train.csv y test.csv
@st.cache_data
def load_data(file_name):
    return pd.read_csv(file_name)

train = load_data('clean_train.csv')
test = load_data('clean_test.csv')

# Escalador
scaler = StandardScaler()
scaler.fit(train.drop(columns=['moid']))

# Función para realizar predicciones
def predict_collision(data, model, scaler):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction

# Formulario para la entrada de datos del usuario
def user_input_features():
    st.sidebar.header("Seleccione las características para la predicción")
    H = st.sidebar.number_input("Magnitud Absoluta (H)", min_value=0.0, max_value=100.0, value=16.9, step=0.1)
    a = st.sidebar.number_input("Eje Semi-Mayor (a)", min_value=0.0, max_value=100.0, value=3.066, step=0.001)
    q = st.sidebar.number_input("Distancia del Perihelio (q)", min_value=0.0, max_value=100.0, value=2.390, step=0.001)
    ad = st.sidebar.number_input("Distancia del Afelio (ad)", min_value=0.0, max_value=100.0, value=3.843, step=0.001)
    n = st.sidebar.number_input("Movimiento Medio (n)", min_value=0.0, max_value=10.0, value=0.24, step=0.001)
    tp_cal = st.sidebar.number_input("Tiempo de Paso por el Perihelio (tp_cal)", min_value=0.0, max_value=1e8, value=2.019578e7, step=1.0)
    per_y = st.sidebar.number_input("Período Orbital (per_y)", min_value=0.0, max_value=1e8, value=47.82621, step=0.01)
    class_options = [str(i) for i in range(13)]  # Creando las opciones de la clase como cadenas
    class_n = st.sidebar.selectbox("Clasificación del Asteroide (class_n)", options=class_options, index=0)
    
    data = {
        'H': H,
        'a': a,
        'q': q,
        'ad': ad,
        'n': n,
        'tp_cal': tp_cal,
        'per_y': per_y,
        'class_n': int(class_n)  # Convertir la selección a entero
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Definir el contenido de cada página
def main_page():
    st.title("Peligrosidad de los asteroides")
    st.write("¿Sabías que la peligrosidad de un asteroide en la Tierra depende de variables distintas a su diámetro?")
    st.write("Cada poco tiempo vemos en diversos medios de comunicación noticias sobre asteroides potencialmente peligrosos para los terrícolas, pero…¿Son estás noticias falsas?¿De qué depende que un asteroide nos haga daño?")
    st.write("Estas preguntas nos ha llevado a querer estudiar estos cuerpos rocosos. Quizás con las respuestas a estas preguntas podamos salvar a la humanidad...¡Quién sabe!")
    st.write("Sobre los asteroides")

def page1():
    st.title("Estadísticas de los Datos")
    
    st.subheader("Resumen Estadístico")
    st.write(train.describe())
    
    st.subheader("Distribución de Variables")
    var = st.selectbox("Selecciona una variable para visualizar su distribución", train.columns)
    
    fig, ax = plt.subplots()
    sns.histplot(train[var], bins=20, kde=True, ax=ax)
    ax.set_title(f'Distribución de {var}')
    st.pyplot(fig)

def page3():
    st.title("Calculadora de Predicción de Colisión de Asteroide")
    input_df = user_input_features()

    # Mostrar los datos de entrada
    st.subheader("Características de Entrada")
    st.write(input_df)

    # Predicción
    if st.button("Predecir Colisión"):
        with st.spinner('Realizando predicción...'):  # Muestra un spinner mientras se realiza la predicción
            time.sleep(2)  # Simulación de proceso de predicción
            prediction = predict_collision(input_df, model, scaler)
        
        st.subheader("Resultado de la Predicción")
        if prediction[0] <= 0.05:  # Asumiendo que un MOID <= 0.05 indica un posible impacto
            st.error("¡Advertencia! El asteroide podría colisionar con la Tierra.")
        else:
            st.success("Es poco probable que el asteroide colisione con la Tierra.")

# Llamar a la función correspondiente a la página seleccionada
if menu == '¿Sabías qué...?':
    main_page()
elif menu == 'Estadísticas de los Datos':
    page1()
elif menu == 'Calculadora de Predicción de Colisión de Asteroide':
    page3()