# Clasificación de Dígitos MNIST con Deep Learning (Unidad 2 - Big Data)

Proyecto integral de Deep Learning para la materia **Big Data (Unidad 2)** de la **Universidad Técnica del Norte (UTN)**.

**Profesor:** Ing. Iván García S., PhD  
**Evaluación:** Diapositiva 22 de `Unidad-2 Big Data.pdf`

---

## 🌟 Modos de Interacción de la Aplicación Web

La aplicación cuenta con **4 formas interactivas de capturar y evaluar dígitos**:

1. 🎨 **Pintar en Cuadro Negro (Canvas):** Dibuja números con el ratón o el dedo sobre un lienzo negro con trazo blanco (como el formato original de MNIST).
2. 📷 **Captura en Vivo con Cámara Web (con Enfoque ROI):** Visor compacto con recuadro verde de enfoque central (Region of Interest - ROI ajustable). El sistema recorta automáticamente solo el área enfocada del papel, eliminando el fondo, manos o sombras.
3. 📁 **Subida de Archivos:** Carga imágenes locales en formato PNG, JPG o BMP.
4. 🔢 **Muestras Oficiales de MNIST:** Banco de pruebas instantáneo con dígitos de prueba del 0 al 9.
5. ⚙️ **Selector de Modelo en Vivo:** Puedes alternar en la barra lateral entre **LeNet-5 (Clásica)** y la **CNN Moderna (Dropout)** para comparar cómo responde cada una.

---

## 📊 1. Resumen de Comparación de Arquitecturas (Diapositiva 22)

Se entrenaron y evaluaron ambas redes sobre las 70.000 imágenes de MNIST durante 10 épocas con optimizador Adam:

| Criterio / Métrica | LeNet-5 (Clásica - Slide 20) | CNN Moderna (Dropout - Slide 21) | Conclusión / Criterio |
| :--- | :---: | :---: | :--- |
| **Accuracy (%)** | **98.47%** | **99.24%** | La CNN Moderna supera por **+0.77%**. |
| **# Parámetros** | **44,426** | **421,642** | LeNet-5 tiene casi 10 veces menos parámetros. |
| **Tamaño del modelo (MB)** | **0.549 MB** | **4.866 MB** | LeNet-5 es casi 9 veces más ligera en disco. |
| **Tiempo de entrenamiento** | **39.09 s** | **138.81 s** (2m 18s) | LeNet-5 entrena 3.5 veces más rápido. |
| **Tiempo de inferencia** | **18.97 ms/img** | **18.03 ms/img** | Latencia prácticamente idéntica en CPU (~18 ms). |

---

## 🚀 2. Cómo Ejecutar Localmente

En tu terminal de PowerShell dentro de esta carpeta:

```powershell
# Iniciar la aplicación web con Streamlit:
.\.venv\Scripts\streamlit.exe run app.py
```

Tu navegador abrirá automáticamente la dirección:  
👉 `http://localhost:8501`

---

## 🌐 3. Despliegue en la Nube (Render.com)

1. Sube tu código a un repositorio de GitHub:
   ```bash
   git init
   git add app.py requirements.txt models/ outputs/ samples/ README.md .gitignore
   git commit -m "feat: MNIST interactive classifier con canvas y camara"
   git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   git push -u origin main
   ```
2. Entra a [Render.com](https://render.com) y crea un **Web Service**.
3. Selecciona tu repositorio y llena los campos:
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   * **Instance Type:** `Free`
