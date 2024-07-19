
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
from PIL import Image

# Configurar la página Streamlit
st.set_page_config(page_title="Mi Aplicación", page_icon=":bar_chart:", layout="wide")

# CSS para la imagen de fondo
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://raw.githubusercontent.com/PazReumante/asteroid_prediction/main/Project/images/imagen.jpeg");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''

# Cargar el CSS en Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)

# Crear un menú de navegación en la barra lateral de la app:
menu = st.sidebar.selectbox('Seleccione una página:', ['¿Sabías qué...?', 'Exploración de datos', 'Calculadora de Predicción de Colisión de Asteroide'])

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return joblib.load('moid_model.pkl')

model = load_model()


# Cargar los datos desde train.csv y test.csv:
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
    H = st.sidebar.number_input("Magnitud Absoluta (H)", min_value=0.0, max_value=100.0, value=22.0, step=0.1)
    a = st.sidebar.number_input("Eje Semi-Mayor (a)", min_value=0.0, max_value=100.0, value=1.1, step=0.001)
    q = st.sidebar.number_input("Distancia del Perihelio (q)", min_value=0.0, max_value=100.0, value=0.9, step=0.001)
    ad = st.sidebar.number_input("Distancia del Afelio (ad)", min_value=0.0, max_value=100.0, value=1.3, step=0.001)
    n = st.sidebar.number_input("Movimiento Medio (n)", min_value=0.0, max_value=10.0, value=0.9, step=0.001)
    tp_cal = st.sidebar.number_input("Tiempo de Paso por el Perihelio (tp_cal)", min_value=0.0, max_value=1e8, value=2458800.5, step=1.0)
    per_y = st.sidebar.number_input("Período Orbital (per_y)", min_value=0.0, max_value=1e8, value=1.1, step=0.01)
    class_options = [str(i) for i in range(13)]  # Creando las opciones de la clase como cadenas
    class_n = st.sidebar.selectbox("Clasificación del Asteroide (class_n)", options=class_options, index=10)
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

    st.markdown(
    """
    <div style="background-color:rgba(0, 0, 0, 0.7); color:white; padding:10px; border-radius:5px;">
    ¿Sabías que la peligrosidad de un asteroide en la Tierra depende de variables distintas a su diámetro?
    </div>
    """, 
    unsafe_allow_html=True)
 
    st.markdown(
    """
    <div style="background-color:rgba(0, 0, 0, 0.7); color:white; padding:10px; border-radius:5px;">
    Cada poco tiempo vemos en diversos medios de comunicación noticias sobre asteroides potencialmente peligrosos para los terrícolas, pero…¿Son estás noticias falsas? ¿De qué depende que un asteroide nos haga daño?
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
    """
    <div style="background-color:rgba(0, 0, 0, 0.7); color:white; padding:10px; border-radius:5px;">  
    Estas preguntas nos ha llevado a querer estudiar estos cuerpos rocosos. Quizás con las respuestas a estas preguntas podamos salvar a la humanidad...¡Quién sabe!")
     </div>
    """, unsafe_allow_html=True)

    st.markdown(
    """
    <div style="background-color:rgba(0, 0, 0, 0.7); color:white; padding:10px; border-radius:5px;">   
    <h2> Sobre los asteroides: </h2>

    Son pequeños objetos rocosos que orbitan alrededor del Sol. La mayoría son más pequeños que los planetas, aunque algunos pueden tener cientos de kilómetros de diámetro. Se forman en diferentes lugares y a diferentes distancias del Sol, además su forma es irregular. El estudio de la composición de estas rocas espaciales nos da información de la historia de los planetas y del Sol.
    
    """, unsafe_allow_html=True)

    st.markdown(
    """
    <div style="background-color:rgba(0, 0, 0, 0.7); color:white; padding:10px; border-radius:5px;">
        <h2>Curiosidades:</h2>
        <ul>
            <li>Varias misiones espaciales de la NASA han volado y observado asteroides. La nave espacial NEAR Shoemaker aterrizó en Eros, un asteroide cerca de la Tierra, en 2001.</li>
            <li>La nave espacial Dawn viajó al cinturón de asteroides en 2011. Orbitó y estudió el asteroide gigante Vesta y el planeta enano Ceres.</li>
            <li>En 2016, la NASA lanzó la nave espacial OSIRIS-REx para estudiar un asteroide cerca de la Tierra llamado Bennu. Después de estudiar a Bennu durante unos años, OSIRIS-REx recogió una muestra de polvo y rocas de la superficie del asteroide. OSIRIS-REx regresó a la Tierra en septiembre de 2023. Actualmente se estudia el polvo y las rocas que la nave recolectó.</li>
        </ul>
    </div>
    """, 
    unsafe_allow_html=True)

    st.markdown(
    """
    <div style="background-color:rgba(0, 0, 0, 0.7); color:#808080; padding:10px; border-radius:5px;">
        <p><a href="https://spaceplace.nasa.gov/asteroid/sp/" style="color:;">Bibliografía</a></p>
    </div>
    """, 
    unsafe_allow_html=True)



def page1():
    st.title("Exploración de los datos")
    st.write("Partimos de una base de datos que contiene 45 características de 958524 asteroides. Al ser una base de datos tan grande, hemos cogido una muestra aleatoria de 150000 asteroides.")
    st.write("¿Qué hemos visto?")
    
    st.write("- La mayoría de asteroides no pasan cerca de la Tierra, siendo neo el indicador si un objeto es cercano a la tierra, donde *Y refleja que el objeto está cercano a la tierra y N* que no existe riesgo de colisión.")
    image_url = "https://raw.githubusercontent.com/PazReumante/asteroid_prediction/main/Project/images/neo.png"
    st.image(image_url, caption="Imagen 1. Histograma neo-número de asteroides.", width=500)

    st.write("- La mayoría de ellos son del cinturón de asteroides que se encuentra entre Marte y Júpiter")
    image_url = "https://raw.githubusercontent.com/PazReumante/asteroid_prediction/main/Project/images/class.png"
    st.image(image_url, caption="Imagen 2. Histograma clase del asteroide-número de asteroides", width=500)

    st.write("- La mayoría de asteroides tienen un diámetro pequeño (<25 km) (adjuntar imagen)")
    image_url = "https://raw.githubusercontent.com/PazReumante/asteroid_prediction/main/Project/images/diameter.png"
    st.image(image_url, caption="Imagen 3. Histograma diámetro-números de asteroides", width=500)

    st.write("Entonces, ¿de qué depende la peligrosidad de un asteroide?")
    
    st.write("""
             
    Para llegar a esta conclusión hemos elegido una variable objetivo llamada Moid. Es una variable que mide la distancia más cercana a la órbita de la Tierra. 
    Relacionando el resto de variables con ella y ayudándonos de una matriz de correlación (Imagen 4), hemos sacado las siguientes conclusiones: 
    - La distancia promedio al Sol (a) es la variable que mas influye en la distancia mas cercana a la Tierra (moid).
    - La distancia máxima al sol (q), la máxima aproximación al Sol (q) , el tiempo que tarda en completar una órbita en años (per_y), la fecha de paso por el Sol (tp_cal) y la procedencia del asteroide (class_n) son variables que también influyen en la distancia mas cercana a la Tierra (moid)""")
    
    st.write("Todas estas conclusiones tienen sentido ya que la gravedad del Sol influye en la trayectoria de estos objetos. Como se puede visualizar en la siguiente matrix de correlación:")

    image_url = "https://raw.githubusercontent.com/PazReumante/asteroid_prediction/main/Project/images/corr_matrix.png"
    st.image(image_url, caption="Imagen 4. Matriz de correlación.", width=500)

    st.write("""
                   **¿Qué problemas hemos encontrado?**
    - Mucha cantidad de datos faltantes , por ejemplo, el diámetro y el albedo (propiedad que tiene cualquier cuerpo de reflejar la radiación que incide sobre él) del objeto. Al faltar el 85% de los datos y tener baja correlación con la variable objetivo, hemos prescindido de ellas.
             """)
    
    st.info("""
    - Visita el repositorio completo en el siguiente enlace:
        [Repositorio](https://github.com/PazReumante/asteroid_prediction)
    - Dataset completo
        [Haz click aquí](https://gitlab.com/mirsakhawathossain/pha-ml/-/raw/master/Dataset/dataset.csv)
        """)

def page3():
    st.title("Calculadora de Predicción de Colisión de Asteroide")

# Definir descripciones de las variables
    descriptions = {
    'H': 'Magnitud absoluta: Imagina que *Asteroide X* está en un escenario brillante en el espacio, y la magnitud absoluta te dice cuánto brilla realmente, como si estuviera a 1 unidad astronómica (UA) del Sol y 1 UA de la Tierra. Es como si todos los asteroides tuvieran que brillar igual para ser comparados en un concurso de luces.',
    
    'a': 'Semieje mayor: Este es el "tamaño" promedio de la pista de carreras por la que *Asteroide X* viaja alrededor del Sol. Imagina una pista ovalada y el semieje mayor es la distancia más larga de un extremo al otro.',
    
    'q': 'Distancia al perihelio: Es como el "punto más cercano" en la pista de carreras de *Asteroide X* al Sol. Piensa en esto como el lugar en la pista donde el asteroide se acerca más al sol durante su recorrido.',
    
    'ad': 'Distancia al afelio: En contraste con el perihelio, el afelio es el "punto más lejano" en la pista de carreras de *Asteroide X* al Sol. Aquí es donde el asteroide se aleja al máximo durante su viaje.',
    
    'n': 'Movimiento medio: Este número te dice cuán rápido va *Asteroide X* en promedio en su pista de carreras. Si *Asteroide X* fuera un corredor, el movimiento medio sería su velocidad promedio a lo largo de la pista.',
    
    'tp_cal': 'Tiempo del paso por el perihelio: Es como el "momento exacto" en el que *Asteroide X* pasa por el punto más cercano al Sol. Imagínate un reloj espacial que te dice cuándo llega a ese punto específico en su pista.',
    
    'per_y': 'Período orbital: Este es el tiempo que tarda *Asteroide X* en dar una vuelta completa alrededor del Sol. Es como la duración de la carrera completa en la pista, medida en años.',
    
    'class_n': 'Tipo de asteroide, con ID numérico: Es como una etiqueta para *Asteroide X*, diciendo qué tipo de asteroide es y dándole un número para clasificarlo. Por ejemplo, puede ser un tipo "A" o "B" con un ID que ayuda a los astrónomos a identificarlo fácilmente.'
}

#Crear 8 columnas para organizar los botones de manera horizontal
    cols = st.columns(8)

    # Crear los 8 botones dentro de las columnas
    for i, var in enumerate(descriptions):
        if cols[i].button(f"{var}"):
            st.write(f"**{var}**: {descriptions[var]}")

    input_df = user_input_features()
    # Mostrar los datos de entrada
    st.subheader("Características de Entrada")
    st.write("""Estos son los parametros seleccionados, te deseamos suerte!""")
    st.write(input_df)
    
# Predicción
    if st.button("Predecir Colisión"):
        prediction = predict_collision(input_df, model, scaler)
        with st.spinner('Realizando predicción...'):  # Muestra un spinner mientras se realiza la predicción
            time.sleep(2)  # Simulación de proceso de predicción
            
        
        if prediction[0] <= 0.05:  # Asumiendo que un MOID <= 0.05 indica un posible impacto
            st.markdown(
        """
        <div style='background-color:#f8d7da;padding:10px;border-radius:5px;color:#721c24;font-weight:bold;'>
        ¡ADVERTENCIA DE COLISIÓN! 🚨: EL ASTEROIDE PUEDE COLISIONAR CON LA TIERRA  ¡TOMA ACCIÓN INMEDIATA! ⚠️
        </div>
        """, unsafe_allow_html=True)
        else:
            st.markdown(
        """
        <div style='background-color:#d4edda;padding:10px;border-radius:5px;color:#155724;font-weight:bold;'>
        Es poco probable que el asteroide colisione con la Tierra.
        </div>
        """, unsafe_allow_html=True)


# Llamar a la función correspondiente a la página seleccionada
if menu == '¿Sabías qué...?':
    main_page()
elif menu == 'Exploración de datos':
    page1()
elif menu == 'Calculadora de Predicción de Colisión de Asteroide':
    page3()
