
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

RUTA_ENTRENAMIENTO = "data/processed/entrenamiento.csv"

VARIABLE_OBJETIVO = "consumo_real"
VARIABLE_GRUPO = "codigo"

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

VARIABLES_PREDICTORAS = (
    VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS
)

# ============================================================
# 2. CARGAR Y FILTRAR DATOS
# ============================================================

datos = pd.read_csv(RUTA_ENTRENAMIENTO)

validos = (
    (datos["promedio"] > 0)
    & (datos["consumo_ajustado_negativo"] == 0)
    & (datos["consumo_imputado_faltante"] == 0)
)

datos = datos.loc[validos].copy()

print("=" * 70)
print("VALIDACIÓN CRUZADA - SIPCAE")
print("=" * 70)

print(f"\nRegistros evaluables: {len(datos)}")
print(f"Usuarios evaluables: {datos[VARIABLE_GRUPO].nunique()}")

# ============================================================
# 3. PREPARAR VARIABLES
# ============================================================

X = datos[VARIABLES_PREDICTORAS].copy()
y = datos[VARIABLE_OBJETIVO].copy()
grupos = datos[VARIABLE_GRUPO].copy()

for columna in VARIABLES_CATEGORICAS:
    X[columna] = X[columna].astype(str)

# ============================================================
# 4. DEFINIR VALIDACIÓN CRUZADA
# ============================================================

validacion = GroupKFold(n_splits=5)

print("\nVALIDACIÓN DE GRUPOS")
print("-" * 45)

for numero, (idx_train, idx_val) in enumerate(
    validacion.split(X, y, groups=grupos), start=1
):
    grupos_train = set(grupos.iloc[idx_train])
    grupos_val = set(grupos.iloc[idx_val])

    compartidos = grupos_train.intersection(grupos_val)

    print(
        f"Partición {numero}: "
        f"entrenamiento={len(idx_train)}, "
        f"validación={len(idx_val)}, "
        f"usuarios compartidos={len(compartidos)}"
    )

    if compartidos:
        raise ValueError(
            "Se encontraron usuarios compartidos entre grupos."
        )

# ============================================================
# 5. DEFINIR PREPROCESAMIENTO
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
# 6. DEFINIR MODELOS
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

# ============================================================
# 7. CONFIGURAR MÉTRICAS
# ============================================================

metricas = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

resultados = []

# ============================================================
# 8. EJECUTAR VALIDACIÓN CRUZADA
# ============================================================

for nombre, modelo in modelos.items():

    print("\n" + "=" * 70)
    print(f"MODELO: {nombre}")
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("modelo", modelo)
        ]
    )

    evaluacion = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        groups=grupos,
        cv=validacion,
        scoring=metricas,
        return_train_score=False,
        n_jobs=1
    )

    for i in range(5):

        mae = -evaluacion["test_MAE"][i]
        rmse = -evaluacion["test_RMSE"][i]
        r2 = evaluacion["test_R2"][i]

        resultados.append({
            "Modelo": nombre,
            "Particion": i + 1,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(
            f"Partición {i + 1}: "
            f"MAE={mae:.2f} | "
            f"RMSE={rmse:.2f} | "
            f"R2={r2:.4f}"
        )

    print("\nPROMEDIO Y DESVIACIÓN ESTÁNDAR")

    print(
        f"MAE:  {-evaluacion['test_MAE'].mean():.2f} "
        f"+/- {evaluacion['test_MAE'].std():.2f}"
    )

    print(
        f"RMSE: {-evaluacion['test_RMSE'].mean():.2f} "
        f"+/- {evaluacion['test_RMSE'].std():.2f}"
    )

    print(
        f"R2:   {evaluacion['test_R2'].mean():.4f} "
        f"+/- {evaluacion['test_R2'].std():.4f}"
    )

# ============================================================
# 9. EVALUAR MODELO DE REFERENCIA
# ============================================================

print("\n" + "=" * 70)
print("MODELO DE REFERENCIA: PROMEDIO HISTÓRICO")
print("=" * 70)

resultados_base = []

for numero, (idx_train, idx_val) in enumerate(
    validacion.split(X, y, groups=grupos), start=1
):

    y_real = y.iloc[idx_val].to_numpy()
    prediccion_base = np.maximum(
        X.iloc[idx_val]["promedio"].to_numpy(),
        0
    )

    mae = mean_absolute_error(y_real, prediccion_base)
    rmse = np.sqrt(
        mean_squared_error(y_real, prediccion_base)
    )
    r2 = r2_score(y_real, prediccion_base)

    resultados_base.append({
        "Modelo": "Promedio histórico",
        "Particion": numero,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(
        f"Partición {numero}: "
        f"MAE={mae:.2f} | "
        f"RMSE={rmse:.2f} | "
        f"R2={r2:.4f}"
    )

resultados.extend(resultados_base)

# ============================================================
# 10. CONSOLIDAR RESULTADOS
# ============================================================

tabla = pd.DataFrame(resultados)

resumen = (
    tabla.groupby("Modelo")[["MAE", "RMSE", "R2"]]
    .agg(["mean", "std"])
)

print("\n" + "=" * 70)
print("RESUMEN GENERAL DE VALIDACIÓN CRUZADA")
print("=" * 70)

print(resumen.round(4))

# ============================================================
# 11. GUARDAR RESULTADOS
# ============================================================

os.makedirs("data/processed", exist_ok=True)

tabla.to_csv(
    "data/processed/resultados_validacion_cruzada.csv",
    index=False
)

resumen.to_csv(
    "data/processed/resumen_validacion_cruzada.csv"
)

print("\nARCHIVOS GENERADOS:")
print("- data/processed/resultados_validacion_cruzada.csv")
print("- data/processed/resumen_validacion_cruzada.csv")

print("\nVALIDACIÓN CRUZADA FINALIZADA.")
