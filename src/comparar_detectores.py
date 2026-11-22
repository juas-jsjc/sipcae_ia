
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

RUTA_DATOS = "data/processed/prueba.csv"
RUTA_MODELO = "models/mejor_modelo_sipcae.joblib"

UMBRAL_INFERIOR = 0.50
UMBRAL_SUPERIOR = 1.50

# ============================================================
# 2. CARGAR DATOS Y MODELO
# ============================================================

df = pd.read_csv(RUTA_DATOS)

modelo = joblib.load(RUTA_MODELO)

# Excluir registros no válidos
validos = (
    (df["promedio"] > 0)
    & (df["consumo_ajustado_negativo"] == 0)
    & (df["consumo_imputado_faltante"] == 0)
)

datos = df.loc[validos].copy()

variables = [
    "promedio",
    "dias_facturados",
    "factor",
    "estrato",
    "desc_clase_servicio",
    "periodo"
]

X = datos[variables].copy()

for columna in ["estrato", "desc_clase_servicio", "periodo"]:
    X[columna] = X[columna].astype(str)

# ============================================================
# 3. CALCULAR CONSUMOS ESPERADOS
# ============================================================

# Referencia histórica
datos["esperado_historico"] = datos["promedio"]

# Predicción del modelo
datos["esperado_modelo"] = np.maximum(
    modelo.predict(X),
    0
)

# ============================================================
# 4. CLASIFICAR ANOMALÍAS
# ============================================================

def clasificar(consumo, esperado):
    if consumo < esperado * UMBRAL_INFERIOR:
        return "Anomalia baja"

    if consumo > esperado * UMBRAL_SUPERIOR:
        return "Anomalia alta"

    return "Normal"


datos["resultado_historico"] = [
    clasificar(consumo, esperado)
    for consumo, esperado in zip(
        datos["consumo_real"],
        datos["esperado_historico"]
    )
]

datos["resultado_modelo"] = [
    clasificar(consumo, esperado)
    for consumo, esperado in zip(
        datos["consumo_real"],
        datos["esperado_modelo"]
    )
]

# ============================================================
# 5. COMPARAR RESULTADOS
# ============================================================

print("=" * 65)
print("COMPARACIÓN DE DETECTORES - SIPCAE")
print("=" * 65)

print("\nDETECTOR BASADO EN PROMEDIO HISTÓRICO:")
print(datos["resultado_historico"].value_counts())

print("\nDETECTOR BASADO EN GRADIENT BOOSTING:")
print(datos["resultado_modelo"].value_counts())

print("\nCOMPARACIÓN DE CLASIFICACIONES:")
comparacion = pd.crosstab(
    datos["resultado_historico"],
    datos["resultado_modelo"],
    rownames=["Promedio histórico"],
    colnames=["Gradient Boosting"]
)

print(comparacion)

# ============================================================
# 6. GUARDAR RESULTADOS
# ============================================================

datos.to_csv(
    "data/processed/comparacion_detectores.csv",
    index=False
)

print("\nArchivo generado:")
print("data/processed/comparacion_detectores.csv")

print("\nCOMPARACIÓN FINALIZADA.")
