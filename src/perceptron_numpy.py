
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Cargar los datos etiquetados
# --------------------------------------------------

train = pd.read_csv("data/processed/entrenamiento_etiquetado.csv")
test = pd.read_csv("data/processed/prueba_etiquetada.csv")

# Variables predictoras
variables = [
    "promedio",
    "dias_facturados",
    "factor",
    "estrato",
    "desc_clase_servicio",
    "periodo"
]

categoricas = ["estrato", "desc_clase_servicio", "periodo"]
numericas = ["promedio", "dias_facturados", "factor"]

X_train = train[variables].copy()
X_test = test[variables].copy()

y_train = train["anomalia"].astype(int).to_numpy()
y_test = test["anomalia"].astype(int).to_numpy()

# --------------------------------------------------
# 2. Codificar variables categóricas
# --------------------------------------------------

# Se unifican las categorías para garantizar
# que entrenamiento y prueba tengan las mismas columnas.
X_completo = pd.concat([X_train, X_test], ignore_index=True)

X_completo = pd.get_dummies(
    X_completo,
    columns=categoricas,
    dtype=float
)

X_train = X_completo.iloc[:len(train)].copy()
X_test = X_completo.iloc[len(train):].copy()

# --------------------------------------------------
# 3. Normalizar variables numéricas
# --------------------------------------------------

# Los parámetros se calculan únicamente con entrenamiento.
medias = X_train[numericas].mean()
desviaciones = X_train[numericas].std().replace(0, 1)

X_train[numericas] = (
    X_train[numericas] - medias
) / desviaciones

X_test[numericas] = (
    X_test[numericas] - medias
) / desviaciones

# Convertir los datos a matrices NumPy
X_train = X_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)

# --------------------------------------------------
# 4. Implementar el perceptrón desde cero
# --------------------------------------------------

np.random.seed(42)

pesos = np.zeros(X_train.shape[1])
sesgo = 0.0

tasa_aprendizaje = 0.01
epocas = 100
peso_anomalia = 2.5

for epoca in range(epocas):

    # Mezclar los registros en cada época
    indices = np.random.permutation(len(X_train))

    for i in indices:

        entrada = X_train[i]
        valor_real = y_train[i]

        # Suma ponderada
        suma = np.dot(entrada, pesos) + sesgo

        # Función de activación escalón
        prediccion = 1 if suma >= 0 else 0

        # Error de clasificación
        error = valor_real - prediccion

        # Dar mayor importancia a las anomalías
        factor_peso = (
            peso_anomalia if valor_real == 1 else 1.0
        )

        # Actualizar pesos y sesgo
        pesos += (
            tasa_aprendizaje
            * error
            * factor_peso
            * entrada
        )

        sesgo += (
            tasa_aprendizaje
            * error
            * factor_peso
        )

# --------------------------------------------------
# 5. Realizar predicciones
# --------------------------------------------------

salidas = np.dot(X_test, pesos) + sesgo
y_pred = (salidas >= 0).astype(int)

# --------------------------------------------------
# 6. Evaluar el modelo
# --------------------------------------------------

verdaderos_negativos = np.sum((y_test == 0) & (y_pred == 0))
falsos_positivos = np.sum((y_test == 0) & (y_pred == 1))
falsos_negativos = np.sum((y_test == 1) & (y_pred == 0))
verdaderos_positivos = np.sum((y_test == 1) & (y_pred == 1))

exactitud = np.mean(y_test == y_pred)

precision = (
    verdaderos_positivos /
    (verdaderos_positivos + falsos_positivos)
    if (verdaderos_positivos + falsos_positivos) > 0 else 0
)

recall = (
    verdaderos_positivos /
    (verdaderos_positivos + falsos_negativos)
    if (verdaderos_positivos + falsos_negativos) > 0 else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

sensibilidad_normal = (
    verdaderos_negativos /
    (verdaderos_negativos + falsos_positivos)
    if (verdaderos_negativos + falsos_positivos) > 0 else 0
)

balanced_accuracy = (recall + sensibilidad_normal) / 2

# --------------------------------------------------
# 7. Mostrar resultados
# --------------------------------------------------

print("\n========================================")
print("      RESULTADOS DEL PERCEPTRÓN")
print("========================================")

print(f"Épocas: {epocas}")
print(f"Variables de entrada: {X_train.shape[1]}")
print(f"Registros de entrenamiento: {len(y_train)}")
print(f"Registros de prueba: {len(y_test)}")

print("\nMATRIZ DE CONFUSIÓN")
print("----------------------------------------")
print("                       Predicción")
print("                    Normal   Anomalía")
print(f"Real Normal         {verdaderos_negativos:6d}   {falsos_positivos:8d}")
print(f"Real Anomalía       {falsos_negativos:6d}   {verdaderos_positivos:8d}")

print("\nMÉTRICAS")
print("----------------------------------------")
print(f"Exactitud (Accuracy):       {exactitud:.4f}")
print(f"Precisión (Precision):      {precision:.4f}")
print(f"Sensibilidad (Recall):      {recall:.4f}")
print(f"F1-score:                   {f1:.4f}")
print(f"Exactitud balanceada:       {balanced_accuracy:.4f}")

print("\n========================================")

# --------------------------------------------------
# 8. Guardar los parámetros del perceptrón
# --------------------------------------------------

Path("models").mkdir(exist_ok=True)

np.savez(
    "models/perceptron_numpy.npz",
    pesos=pesos,
    sesgo=sesgo,
    medias=medias.to_numpy(),
    desviaciones=desviaciones.to_numpy()
)

print("Modelo guardado en models/perceptron_numpy.npz")
