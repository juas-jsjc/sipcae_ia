
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

RUTA_ENTRENAMIENTO = "data/processed/entrenamiento.csv"
RUTA_PRUEBA = "data/processed/prueba.csv"

VARIABLE_OBJETIVO = "consumo_real"

VARIABLES_NUMERICAS = [
    "promedio",
    "dias_facturados",
    "factor"
]

VARIABLES_CATEGORICAS = [
    "estrato",
    "desc_clase_servicio",
    "periodo"
]

VARIABLES_PREDICTORAS = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS

VARIABLES_CONTROL = [
    "consumo_ajustado_negativo",
    "consumo_imputado_faltante"
]

# ============================================================
# 2. CARGAR Y FILTRAR DATOS
# ============================================================

entrenamiento = pd.read_csv(RUTA_ENTRENAMIENTO)
prueba = pd.read_csv(RUTA_PRUEBA)


def filtrar_registros(datos):
    validos = (
        (datos["promedio"] > 0)
        & (datos["consumo_ajustado_negativo"] == 0)
        & (datos["consumo_imputado_faltante"] == 0)
    )

    return datos.loc[validos].copy()


entrenamiento = filtrar_registros(entrenamiento)
prueba = filtrar_registros(prueba)

print("=" * 65)
print("ENTRENAMIENTO DE MODELOS DE REGRESIÓN - SIPCAE")
print("=" * 65)

print(f"\nRegistros de entrenamiento: {len(entrenamiento)}")
print(f"Registros de prueba: {len(prueba)}")

# ============================================================
# 3. PREPARAR VARIABLES
# ============================================================

X_train = entrenamiento[VARIABLES_PREDICTORAS]
y_train = entrenamiento[VARIABLE_OBJETIVO]

X_test = prueba[VARIABLES_PREDICTORAS]
y_test = prueba[VARIABLE_OBJETIVO]

# Convertir variables categóricas a texto
for columna in VARIABLES_CATEGORICAS:
    X_train[columna] = X_train[columna].astype(str)
    X_test[columna] = X_test[columna].astype(str)

# ============================================================
# 4. PREPROCESAMIENTO
# ============================================================

preprocesador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            VARIABLES_CATEGORICAS
        ),
        (
            "numericas",
            "passthrough",
            VARIABLES_NUMERICAS
        )
    ]
)

# ============================================================
# 5. DEFINIR MODELOS
# ============================================================

modelos = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        loss="huber",
        random_state=42
    )
}

resultados = {}
modelos_entrenados = {}

# ============================================================
# 6. ENTRENAR Y EVALUAR
# ============================================================

for nombre, modelo in modelos.items():

    print(f"\nEntrenando modelo: {nombre}")

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("modelo", modelo)
        ]
    )

    pipeline.fit(X_train, y_train)

    predicciones = pipeline.predict(X_test)

    # Evitar predicciones negativas de consumo
    predicciones = np.maximum(predicciones, 0)

    mae = mean_absolute_error(y_test, predicciones)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    r2 = r2_score(y_test, predicciones)

    resultados[nombre] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    modelos_entrenados[nombre] = pipeline

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

# ============================================================
# 7. COMPARAR CONTRA EL PROMEDIO HISTÓRICO
# ============================================================

prediccion_base = np.maximum(X_test["promedio"].to_numpy(), 0)

mae_base = mean_absolute_error(y_test, prediccion_base)

print("\n" + "=" * 65)
print("MODELO DE REFERENCIA: PROMEDIO HISTÓRICO")
print("=" * 65)

print(f"MAE del promedio histórico: {mae_base:.2f}")

# ============================================================
# 8. COMPARAR RESULTADOS
# ============================================================

tabla_resultados = pd.DataFrame(resultados).T
tabla_resultados = tabla_resultados.sort_values("MAE")

print("\nCOMPARACIÓN DE MODELOS:")
print(tabla_resultados.round(4))

# Seleccionar el modelo con menor MAE
mejor_modelo_nombre = tabla_resultados.index[0]
mejor_modelo = modelos_entrenados[mejor_modelo_nombre]

print(f"\nMejor modelo: {mejor_modelo_nombre}")

# ============================================================
# 9. GUARDAR MODELO
# ============================================================

os.makedirs("models", exist_ok=True)

ruta_modelo = "models/mejor_modelo_sipcae.joblib"

joblib.dump(mejor_modelo, ruta_modelo)

tabla_resultados.to_csv(
    "data/processed/resultados_modelos.csv"
)

print(f"\nModelo guardado en: {ruta_modelo}")
print("Resultados guardados en: data/processed/resultados_modelos.csv")

print("\nENTRENAMIENTO FINALIZADO.")
