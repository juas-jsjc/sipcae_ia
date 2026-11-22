
import pandas as pd
import numpy as np

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

RUTA_PRUEBA = "data/processed/prueba.csv"
RUTA_RESULTADOS = "data/processed/deteccion_anomalias.csv"

UMBRAL_INFERIOR = 0.50
UMBRAL_SUPERIOR = 1.50

# ============================================================
# 2. CARGAR DATOS
# ============================================================

df = pd.read_csv(RUTA_PRUEBA)

print("=" * 65)
print("DETECCIÓN DE ANOMALÍAS - SIPCAE")
print("=" * 65)

# ============================================================
# 3. FILTRAR REGISTROS VÁLIDOS
# ============================================================

registros_iniciales = len(df)

validos = (
    (df["promedio"] > 0)
    & (df["consumo_ajustado_negativo"] == 0)
    & (df["consumo_imputado_faltante"] == 0)
)

datos = df.loc[validos].copy()

print(f"\nRegistros iniciales: {registros_iniciales}")
print(f"Registros evaluables: {len(datos)}")
print(f"Registros excluidos: {registros_iniciales - len(datos)}")

# ============================================================
# 4. CALCULAR DESVIACIÓN DEL CONSUMO
# ============================================================

datos["consumo_esperado"] = datos["promedio"]

datos["desviacion_porcentual"] = (
    (datos["consumo_real"] - datos["consumo_esperado"])
    / datos["consumo_esperado"]
) * 100

# ============================================================
# 5. CLASIFICAR COMPORTAMIENTOS
# ============================================================

condiciones = [
    datos["consumo_real"] < (
        datos["consumo_esperado"] * UMBRAL_INFERIOR
    ),

    datos["consumo_real"] > (
        datos["consumo_esperado"] * UMBRAL_SUPERIOR
    )
]

categorias = [
    "Anomalia baja",
    "Anomalia alta"
]

datos["resultado"] = np.select(
    condiciones,
    categorias,
    default="Normal"
)

datos["anomalia"] = (
    datos["resultado"] != "Normal"
).astype(int)

# ============================================================
# 6. GENERAR REPORTE
# ============================================================

columnas_reporte = [
    "promedio",
    "consumo_real",
    "consumo_esperado",
    "desviacion_porcentual",
    "resultado",
    "anomalia",
    "estrato",
    "desc_clase_servicio",
    "periodo"
]

reporte = datos[columnas_reporte].copy()

reporte.to_csv(RUTA_RESULTADOS, index=False)

# ============================================================
# 7. MOSTRAR RESULTADOS
# ============================================================

print("\nDISTRIBUCIÓN DE RESULTADOS:")
print(datos["resultado"].value_counts())

print("\nPORCENTAJES:")
print(
    (datos["resultado"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nRESUMEN DE DESVIACIONES:")
print(datos["desviacion_porcentual"].describe().round(2))

print("\nREGISTROS CON MAYORES DESVIACIONES:")
print(
    datos.nlargest(10, "desviacion_porcentual")[
        ["promedio", "consumo_real", "desviacion_porcentual", "resultado"]
    ].to_string(index=False)
)

print("\nARCHIVO GENERADO:")
print(RUTA_RESULTADOS)

print("\nDETECCIÓN FINALIZADA.")
