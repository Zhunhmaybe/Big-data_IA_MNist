"""
Script de Entrenamiento y Comparación Cuantitativa de Modelos para MNIST
Materia: Big Data - Unidad 2 (Deep Learning) - UTN

Compara dos arquitecturas:
1. LeNet-5 (Yann LeCun, Diapositiva 20)
2. CNN Moderna con Dropout (Diapositiva 21)

Calcula las 5 metricas exigidas en la Diapositiva 22:
- Accuracy (%)
- # de parametros a entrenar
- Tamano del modelo (MB)
- Tiempo de entrenamiento: model.fit (s, min)
- Tiempo de inferencia (ms por imagen)
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Asegurar reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Crear directorio de salidas
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess_data():
    print(">>> Cargando el dataset MNIST (70.000 imagenes 28x28)...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalizar valores de pixeles de [0, 255] a [0.0, 1.0]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expandir dimensiones para el canal en escala de grises: (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"Dimensiones de entrenamiento: {x_train.shape}, etiquetas: {y_train.shape}")
    print(f"Dimensiones de prueba (test): {x_test.shape}, etiquetas: {y_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def build_lenet5():
    """
    Arquitectura LeNet-5 adaptada para entrada 28x28 (Diapositiva 20):
    - Conv1: 6 filtros 5x5, ReLU (salida 24x24x6)
    - MaxPool1: 2x2, stride 2 (salida 12x12x6)
    - Conv2: 16 filtros 5x5, ReLU (salida 8x8x16)
    - MaxPool2: 2x2, stride 2 (salida 4x4x16)
    - Flatten: 256
    - Dense1: 120 neuronas, ReLU
    - Dense2: 84 neuronas, ReLU
    - Salida: 10 neuronas, Softmax
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(6, kernel_size=(5, 5), strides=1, padding="valid", activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid", name="pool1"),
        layers.Conv2D(16, kernel_size=(5, 5), strides=1, padding="valid", activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid", name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(120, activation="relu", name="dense_fc1"),
        layers.Dense(84, activation="relu", name="dense_fc2"),
        layers.Dense(10, activation="softmax", name="output_softmax")
    ], name="LeNet-5")
    return model


def build_modern_cnn():
    """
    Arquitectura CNN Moderna con Dropout (Diapositiva 21):
    - Conv1: 32 filtros 3x3, ReLU, padding 'same'
    - MaxPool1: 2x2
    - Conv2: 64 filtros 3x3, ReLU, padding 'same'
    - MaxPool2: 2x2
    - Dropout(0.25) para regularizacion
    - Flatten
    - Dense1: 128 neuronas, ReLU
    - Dropout(0.50) para regularizacion
    - Salida: 10 neuronas, Softmax
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1_32"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2_64"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Dropout(0.25, name="dropout_conv"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_128"),
        layers.Dropout(0.50, name="dropout_dense"),
        layers.Dense(10, activation="softmax", name="output_softmax")
    ], name="Modern_CNN_Dropout")
    return model


def evaluate_inference_latency(model, x_test, num_samples=1000):
    """
    Mide el tiempo promedio de inferencia por imagen en milisegundos.
    """
    # Muestras para el test de latencia
    samples = x_test[:num_samples]
    
    # Warm-up (calentamiento inicial del compilador de TensorFlow)
    _ = model.predict(samples[:10], verbose=0)
    
    # Medir inferencia imagen a imagen (simulando peticiones web individuales)
    start_time = time.perf_counter()
    for i in range(num_samples):
        # Pasar muestra individual (1, 28, 28, 1)
        single_img = samples[i:i+1]
        _ = model(single_img, training=False)
    total_time = time.perf_counter() - start_time
    
    ms_per_image = (total_time / num_samples) * 1000.0
    return ms_per_image, total_time


def train_and_evaluate_model(build_fn, model_name, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    print(f"\n{'='*60}")
    print(f" Iniciando entrenamiento de: {model_name}")
    print(f"{'='*60}")

    model = build_fn()
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 1. Metrica: # de parametros
    total_params = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)

    # 2. Metrica: Tiempo de entrenamiento
    start_train = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    training_time_sec = time.time() - start_train

    # 3. Metrica: Accuracy en el conjunto de prueba
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    accuracy_pct = test_acc * 100.0

    # 4. Metrica: Tamano del modelo en disco (MB)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.keras")
    model.save(model_path)
    model_size_mb = os.path.getsize(model_path) / (1024.0 * 1024.0)

    # 5. Metrica: Tiempo de inferencia
    print(f"> Evaluando latencia de inferencia para {model_name}...")
    latency_ms, total_inf_time = evaluate_inference_latency(model, x_test, num_samples=500)

    results = {
        "model_name": model_name,
        "accuracy_pct": round(float(accuracy_pct), 3),
        "test_loss": round(float(test_loss), 4),
        "trainable_params": int(trainable_params),
        "model_size_mb": round(float(model_size_mb), 3),
        "training_time_sec": round(float(training_time_sec), 2),
        "training_time_formatted": f"{int(training_time_sec // 60)}m {int(training_time_sec % 60)}s",
        "inference_latency_ms": round(float(latency_ms), 3),
        "model_path": model_path,
        "history": {
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
            "accuracy": [float(x) for x in history.history["accuracy"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        }
    }

    return model, results


def plot_learning_curves(results_lenet, results_modern):
    """
    Genera graficos de Loss y Accuracy para ambos modelos.
    """
    epochs_range = range(1, len(results_lenet["history"]["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico 1: Accuracy
    axes[0].plot(epochs_range, [acc * 100 for acc in results_lenet["history"]["accuracy"]], "b--", label="LeNet-5 (Train)")
    axes[0].plot(epochs_range, [acc * 100 for acc in results_lenet["history"]["val_accuracy"]], "b-", label="LeNet-5 (Val)")
    axes[0].plot(epochs_range, [acc * 100 for acc in results_modern["history"]["accuracy"]], "g--", label="CNN Moderna (Train)")
    axes[0].plot(epochs_range, [acc * 100 for acc in results_modern["history"]["val_accuracy"]], "g-", label="CNN Moderna (Val)")
    axes[0].set_title("Exactitud (Accuracy %) durante el entrenamiento", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Grafico 2: Loss
    axes[1].plot(epochs_range, results_lenet["history"]["loss"], "b--", label="LeNet-5 (Train Loss)")
    axes[1].plot(epochs_range, results_lenet["history"]["val_loss"], "b-", label="LeNet-5 (Val Loss)")
    axes[1].plot(epochs_range, results_modern["history"]["loss"], "g--", label="CNN Moderna (Train Loss)")
    axes[1].plot(epochs_range, results_modern["history"]["val_loss"], "g-", label="CNN Moderna (Val Loss)")
    axes[1].set_title("Perdida (Loss) durante el entrenamiento", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUTS_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[OK] Graficas de curvas de aprendizaje guardadas en: {plot_path}")
    return plot_path


def main():
    print("="*70)
    print(" EXPERIMENTO COMPARATIVO: LeNet-5 vs CNN Moderna con Dropout ")
    print("="*70)

    # Cargar datos
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Entrenar Modelo 1: LeNet-5
    model_lenet, res_lenet = train_and_evaluate_model(
        build_lenet5, "LeNet-5",
        x_train, y_train, x_test, y_test,
        epochs=10, batch_size=128
    )

    # Entrenar Modelo 2: CNN Moderna con Dropout
    model_modern, res_modern = train_and_evaluate_model(
        build_modern_cnn, "Modern_CNN_Dropout",
        x_train, y_train, x_test, y_test,
        epochs=10, batch_size=128
    )

    # Guardar curvas comparativas
    plot_learning_curves(res_lenet, res_modern)

    # Determinar el mejor modelo segun Accuracy y Fps/latencia
    if res_modern["accuracy_pct"] >= res_lenet["accuracy_pct"]:
        best_model_name = "Modern_CNN_Dropout"
        best_model = model_modern
    else:
        best_model_name = "LeNet-5"
        best_model = model_lenet

    best_model_path = os.path.join(MODELS_DIR, "best_model.keras")
    best_model.save(best_model_path)
    print(f"\n[OK] Mejor modelo guardado para la aplicacion web en: {best_model_path}")

    # Guardar resumen en JSON
    comparison_data = {
        "lenet_5": res_lenet,
        "modern_cnn": res_modern,
        "best_model_selected": best_model_name
    }
    json_path = os.path.join(OUTPUTS_DIR, "comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=4)

    # Generar tabla comparativa en Markdown
    md_table = f"""# Tabla Comparativa de Modelos (Diapositiva 22)

| Criterio / Metrica | LeNet-5 (Clasica) | CNN Moderna (Dropout) | Diferencia / Observacion |
| :--- | :---: | :---: | :--- |
| **Accuracy (%)** | **{res_lenet['accuracy_pct']}%** | **{res_modern['accuracy_pct']}%** | {'CNN Moderna supera por +' + str(round(res_modern['accuracy_pct'] - res_lenet['accuracy_pct'], 2)) + '%' if res_modern['accuracy_pct'] >= res_lenet['accuracy_pct'] else 'LeNet-5 supera'} |
| **# Parametros a entrenar** | {res_lenet['trainable_params']:,} | {res_modern['trainable_params']:,} | {'LeNet-5 es mas ligera' if res_lenet['trainable_params'] < res_modern['trainable_params'] else 'CNN es mas ligera'} |
| **Tamano del modelo (MB)** | {res_lenet['model_size_mb']} MB | {res_modern['model_size_mb']} MB | Archivo .keras en disco |
| **Tiempo de entrenamiento** | {res_lenet['training_time_formatted']} ({res_lenet['training_time_sec']}s) | {res_modern['training_time_formatted']} ({res_modern['training_time_sec']}s) | 10 epocas con batch de 128 |
| **Tiempo de inferencia (ms/img)** | {res_lenet['inference_latency_ms']} ms | {res_modern['inference_latency_ms']} ms | Latencia por prediccion individual |

**Modelo Seleccionado para el Despliegue Web:** `{best_model_name}`
"""
    md_path = os.path.join(OUTPUTS_DIR, "comparison_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print("\n" + "="*70)
    print(" RESUMEN FINAL DE LA COMPARACION ")
    print("="*70)
    print(md_table)
    print(f"\n[OK] Resultados guardados en: {md_path} y {json_path}")


if __name__ == "__main__":
    main()
