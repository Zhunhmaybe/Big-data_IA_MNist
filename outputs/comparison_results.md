# Tabla Comparativa de Modelos (Diapositiva 22)

| Criterio / Metrica | LeNet-5 (Clasica) | CNN Moderna (Dropout) | Diferencia / Observacion |
| :--- | :---: | :---: | :--- |
| **Accuracy (%)** | **98.47%** | **99.24%** | CNN Moderna supera por +0.77% |
| **# Parametros a entrenar** | 44,426 | 421,642 | LeNet-5 es mas ligera |
| **Tamano del modelo (MB)** | 0.549 MB | 4.866 MB | Archivo .keras en disco |
| **Tiempo de entrenamiento** | 0m 39s (39.09s) | 2m 18s (138.81s) | 10 epocas con batch de 128 |
| **Tiempo de inferencia (ms/img)** | 18.968 ms | 18.033 ms | Latencia por prediccion individual |

**Modelo Seleccionado para el Despliegue Web:** `Modern_CNN_Dropout`
