"""
Aplicacion Web Avanzada para Clasificacion de Digitos MNIST
Materia: Big Data - Unidad 2 (Deep Learning) - UTN
Profesor: Ing. Ivan Garcia S., PhD

Caracteristicas avanzadas:
1. Pintar en lienzo interactivo (Cuadro negro con pincel blanco).
2. Captura en vivo desde camara web (st.camera_input).
3. Carga de archivos de imagen (JPG, PNG).
4. Galeria de muestras reales de prueba (0 al 9).
5. Selector interactivo de modelo: Comparar en vivo LeNet-5 vs CNN Moderna.
6. Pipeline inteligente de centrado y normalizacion espacial (Estandar MNIST).
"""

import os
import json
import time
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Intentar importar canvas interactivo
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# Configuracion de pagina
st.set_page_config(
    page_title="MNIST Digit Classifier | Deep Learning UTN",
    page_icon="🔢",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
COMPARISON_JSON_PATH = os.path.join(OUTPUTS_DIR, "comparison_results.json")


@st.cache_resource
def load_models_dict():
    """
    Carga en memoria los dos modelos entrenados para poder alternar entre ellos.
    """
    import tensorflow as tf
    models = {}
    for name, filename in [("CNN Moderna (Dropout)", "Modern_CNN_Dropout.keras"), ("LeNet-5 (Clasica)", "LeNet-5.keras")]:
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path)
        else:
            # Fallback a best_model si existe
            best_path = os.path.join(MODELS_DIR, "best_model.keras")
            if os.path.exists(best_path):
                models[name] = tf.keras.models.load_model(best_path)
    return models


def center_and_normalize_digit(image: Image.Image, invert_if_light=True) -> tuple[np.ndarray, Image.Image]:
    """
    Pipeline estandar de centrado de MNIST (Yann LeCun):
    1. Convierte a escala de grises.
    2. Invierte si el fondo es claro (papel blanco con tinta negra).
    3. Detecta la caja delimitadora (bounding box) del trazo.
    4. Redimensiona manteniendo relacion de aspecto para caber en una caja de 20x20.
    5. Centra el digito dentro de un lienzo de 28x28 (dejando margen de 4 px).
    6. Normaliza a rango [0.0, 1.0].
    """
    gray = image.convert("L")
    arr = np.array(gray)

    # 1. Detectar si el fondo es claro (foto de papel o fondo blanco)
    if invert_if_light:
        border_mean = np.mean([
            arr[0, :].mean(), arr[-1, :].mean(), arr[:, 0].mean(), arr[:, -1].mean()
        ])
        if border_mean > 120:
            gray = ImageOps.invert(gray)
            arr = np.array(gray)

    # 2. Filtrar ruido tenue
    arr = np.where(arr < 35, 0, arr)

    # 3. Encontrar coordenadas del digito
    coords = np.argwhere(arr > 30)
    if coords.size == 0:
        # Lienzo vacio
        empty_28 = Image.new("L", (28, 28), color=0)
        return np.zeros((1, 28, 28, 1), dtype="float32"), empty_28

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Recortar el digito
    cropped = gray.crop((x_min, y_min, x_max + 1, y_max + 1))
    w, h = cropped.size

    # 4. Escalar para que el lado mayor mida 20 pixeles
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h / w) * 20)))
    else:
        new_h = 20
        new_w = max(1, int(round((w / h) * 20)))

    resized_digit = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 5. Pegar centrado en lienzo negro de 28x28
    final_img = Image.new("L", (28, 28), color=0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    final_img.paste(resized_digit, (paste_x, paste_y))

    # 6. Normalizar
    norm_arr = np.array(final_img, dtype="float32") / 255.0
    tensor = np.expand_dims(norm_arr, axis=(0, -1))
    return tensor, final_img


def main():
    st.title("🔢 Clasificador Inteligente de Dígitos MNIST")
    st.markdown("""
    **Materia:** Big Data — *Unidad 2: Deep Learning (Convoluciones y Regularización)*  
    **Institución:** Universidad Técnica del Norte (UTN)  
    **Evaluación:** Diapositiva 22 — Comparación cuantitativa y despliegue web.
    """)

    st.divider()

    # Cargar modelos
    models_dict = load_models_dict()

    # Barra lateral
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selector de modelo
        model_options = list(models_dict.keys())
        if model_options:
            selected_model_name = st.selectbox(
                "Selecciona el modelo para predecir:",
                model_options,
                index=0
            )
            active_model = models_dict[selected_model_name]
            st.caption(f"Parámetros del modelo: **{active_model.count_params():,}**")
        else:
            st.error("No se encontraron modelos en la carpeta `models/`.")
            active_model = None
            selected_model_name = None

        st.markdown("---")
        st.subheader("📊 Métricas de la Diapositiva 22")
        if os.path.exists(COMPARISON_JSON_PATH):
            try:
                with open(COMPARISON_JSON_PATH, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                
                lenet = comp_data["lenet_5"]
                modern = comp_data["modern_cnn"]
                
                st.markdown(f"""
                | Criterio | LeNet-5 | CNN Moderna |
                | :--- | :---: | :---: |
                | **Accuracy** | `{lenet['accuracy_pct']}%` | `{modern['accuracy_pct']}%` |
                | **Parámetros** | `{lenet['trainable_params']:,}` | `{modern['trainable_params']:,}` |
                | **Tamaño** | `{lenet['model_size_mb']} MB` | `{modern['model_size_mb']} MB` |
                | **Entrenamiento** | `{lenet['training_time_sec']}s` | `{modern['training_time_sec']}s` |
                | **Inferencia** | `{lenet['inference_latency_ms']} ms` | `{modern['inference_latency_ms']} ms` |
                """)
                st.info(f"🏆 Mejor arquitectura: **{comp_data['best_model_selected']}** (+0.77% de accuracy)")
            except Exception as e:
                st.caption(f"Error leyendo comparativa: {e}")

    # Tabs para los diferentes metodos de entrada
    tab_canvas, tab_camera, tab_upload, tab_samples = st.tabs([
        "🎨 Dibujar en Cuadro Negro (Canvas)",
        "📷 Capturar con Cámara Web",
        "📁 Subir Imagen",
        "🔢 Muestras Oficiales MNIST"
    ])

    input_image = None
    invert_needed = True

    # 1. TAB CANVAS
    with tab_canvas:
        st.subheader("Dibuja un dígito con el ratón o el dedo")
        st.caption("Traza un número del 0 al 9 en el cuadro negro con trazo blanco (exactamente como MNIST).")

        col_c1, col_c2 = st.columns([1.5, 1])
        with col_c1:
            if CANVAS_AVAILABLE:
                stroke_width = st.slider("Grosor del pincel:", min_value=10, max_value=28, value=18, step=2)
                canvas_result = st_canvas(
                    fill_color="#000000",
                    stroke_width=stroke_width,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    width=280,
                    height=280,
                    drawing_mode="freedraw",
                    key="canvas_mnist",
                    display_toolbar=True
                )
                if canvas_result.image_data is not None:
                    # Convertir datos RGBA de canvas a PIL Image
                    raw_img = Image.fromarray(canvas_result.image_data.astype("uint8"))
                    # Verificar si el usuario ha dibujado algo
                    if np.max(canvas_result.image_data[:, :, :3]) > 10:
                        input_image = raw_img
                        invert_needed = False  # Ya es fondo negro y trazo blanco
            else:
                st.warning("El módulo de dibujo interactivo no está disponible. Puedes usar las opciones de Cámara o Subir Archivo.")

        with col_c2:
            st.markdown("""
            **Consejos para dibujar:**
            * Dibuja el número en el centro.
            * Usa un trazo firme.
            * Nuestro algoritmo de preprocesamiento centrará tu trazo automáticamente a $20 \\times 20$ dentro de $28 \\times 28$.
            """)

    # 2. TAB CAMARA
    with tab_camera:
        st.subheader("📷 Captura en Vivo con Cámara Web")
        st.markdown("""
        <style>
        /* Reducir y estilizar el contenedor de la camara para que no ocupe toda la pantalla */
        div[data-testid="stCameraInput"] {
            max-width: 360px !important;
            margin: 0 auto;
        }
        div[data-testid="stCameraInput"] video {
            border-radius: 10px;
            border: 2px solid #3b82f6;
            max-height: 240px !important;
            object-fit: cover;
        }
        </style>
        """, unsafe_allow_html=True)

        col_cam_ctrl, col_cam_preview = st.columns([1, 1], gap="medium")

        with col_cam_ctrl:
            st.markdown("""
            **🎯 Instrucciones de enfoque:**
            1. Escribe el número en una hoja de papel (con marcador o tinta oscura).
            2. Centra el número en la cámara dentro del **recuadro de enfoque**.
            3. Haz clic en **Tomar foto**. El sistema recortará automáticamente solo esa área, eliminando el fondo.
            """)
            
            roi_percent = st.slider(
                "Tamaño del recuadro de enfoque (ROI):",
                min_value=30, max_value=80, value=50, step=5,
                help="Ajusta el porcentaje central de la imagen que se extraerá para analizar el dígito."
            )
            roi_ratio = roi_percent / 100.0

            camera_photo = st.camera_input("Capturar foto del dígito", label_visibility="collapsed")

        with col_cam_preview:
            if camera_photo is not None:
                from PIL import ImageDraw
                raw_cam_img = Image.open(camera_photo)
                w, h = raw_cam_img.size

                # Calcular coordenadas del recuadro central (ROI)
                roi_size = int(min(w, h) * roi_ratio)
                x1 = (w - roi_size) // 2
                y1 = (h - roi_size) // 2
                x2 = x1 + roi_size
                y2 = y1 + roi_size

                # Dibujar recuadro de enfoque en verde sobre la foto capturada
                annotated_img = raw_cam_img.copy()
                draw = ImageDraw.Draw(annotated_img)
                # Borde verde visible con esquinas de enfoque
                draw.rectangle([x1, y1, x2, y2], outline="#22c55e", width=4)
                
                # Recortar unicamente la region central de interes (ROI)
                cropped_roi = raw_cam_img.crop((x1, y1, x2, y2))

                st.markdown("**Vista de Captura y Enfoque:**")
                subcol_a, subcol_b = st.columns([1.2, 1])
                with subcol_a:
                    st.image(
                        annotated_img,
                        caption="Foto capturada (Enfoque en verde)",
                        width=220
                    )
                with subcol_b:
                    st.image(
                        cropped_roi,
                        caption="Área de enfoque (ROI recortada)",
                        width=130
                    )

                # La imagen que se pasa al clasificador es solo el recorte del ROI
                input_image = cropped_roi
                invert_needed = True
            else:
                st.info("💡 Activa tu cámara web y presiona 'Tomar foto' para enfocar y predecir.")

    # 3. TAB SUBIR ARCHIVO
    with tab_upload:
        st.subheader("Sube cualquier imagen desde tu computadora")
        uploaded_file = st.file_uploader(
            "Selecciona un archivo de imagen (PNG, JPG, BMP):",
            type=["png", "jpg", "jpeg", "bmp"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            invert_needed = True

    # 4. TAB MUESTRAS REALES MNIST
    with tab_samples:
        st.subheader("Prueba rápida con dígitos reales de MNIST")
        sample_digit = st.selectbox(
            "Elige un dígito oficial del conjunto de prueba (Test Set):",
            list(range(10)),
            key="select_sample_digit"
        )
        sample_path = os.path.join(SAMPLES_DIR, f"mnist_{sample_digit}.png")
        if os.path.exists(sample_path):
            sample_img = Image.open(sample_path)
            st.image(sample_img.resize((140, 140)), caption=f"Muestra oficial: Dígito {sample_digit}", width=140)
            if st.button("Evaluar esta muestra"):
                input_image = sample_img
                invert_needed = False

    st.divider()

    # SECCION DE RESULTADOS Y PREDICCION
    if input_image is not None and active_model is not None:
        st.header("🔮 Resultado de la Inferencia")

        col_res1, col_res2, col_res3 = st.columns([1, 1.2, 1.8], gap="medium")

        # Columna 1: Vista previa
        with col_res1:
            st.markdown("**1. Imagen Capturada:**")
            st.image(input_image, width=160)

        # Columna 2: Preprocesamiento MNIST
        tensor, processed_28x28 = center_and_normalize_digit(input_image, invert_if_light=invert_needed)
        with col_res2:
            st.markdown("**2. Adaptada a $28 \\times 28$ (MNIST):**")
            st.image(processed_28x28.resize((140, 140)), width=140, caption="Entrada real a la red")

        # Columna 3: Prediccion
        with col_res3:
            st.markdown(f"**3. Predicción con `{selected_model_name}`:**")
            
            t_start = time.perf_counter()
            preds = active_model.predict(tensor, verbose=0)[0]
            lat_ms = (time.perf_counter() - t_start) * 1000.0

            predicted_digit = int(np.argmax(preds))
            confidence = float(preds[predicted_digit]) * 100.0

            st.markdown(f"""
            <div style="background-color: #0f172a; padding: 18px; border-radius: 12px; border-left: 6px solid #3b82f6;">
                <h3 style="color: #94a3b8; margin: 0; font-size: 16px;">DÍGITO PREDICHO</h3>
                <h1 style="color: #60a5fa; margin: 5px 0 0 0; font-size: 54px; font-weight: bold;">{predicted_digit}</h1>
                <p style="color: #cbd5e1; margin: 6px 0 0 0;">Confianza: <b>{confidence:.2f}%</b> &nbsp;|&nbsp; Latencia: <b>{lat_ms:.2f} ms</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Grafico de barras de distribucion
        st.markdown("#### Probabilidad para cada Dígito (0 al 9):")
        prob_dict = {f"Dígito {i}": float(preds[i] * 100) for i in range(10)}
        st.bar_chart(prob_dict)

    elif active_model is None:
        st.error("Por favor entrena los modelos antes de probar la inferencia.")
    else:
        st.info("👆 Dibuja un número en el cuadro negro, toma una foto con tu cámara web o sube un archivo para ver la predicción.")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 12px;">
        Desarrollado para la materia Big Data - Universidad Técnica del Norte (UTN) | Modelos: LeNet-5 y CNN Moderna
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
