
import pandas as pd

# Cargar resultados del detector
df = pd.read_csv("data/processed/deteccion_anomalias.csv")

print("=" * 65)
print("ANÁLISIS DE ANOMALÍAS - SIPCAE")
print("=" * 65)

# 1. Distribución por clase de servicio
print("\n1. ANOMALÍAS POR CLASE DE SERVICIO")

tabla_clase = pd.crosstab(
    df["desc_clase_servicio"],
    df["resultado"]
)

print(tabla_clase)

# 2. Distribución por periodo
print("\n2. ANOMALÍAS POR PERIODO")

tabla_periodo = pd.crosstab(
    df["periodo"],
    df["resultado"]
)

print(tabla_periodo)

# 3. Porcentaje de anomalías por clase
print("\n3. PORCENTAJE DE ANOMALÍAS POR CLASE")

porcentaje_clase = (
    df.assign(anomalia=df["anomalia"])
    .groupby("desc_clase_servicio")["anomalia"]
    .mean() * 100
)

print(porcentaje_clase.round(2))

# 4. Diez anomalías con mayor desviación positiva
print("\n4. DIEZ MAYORES DESVIACIONES POSITIVAS")

mayores = df.nlargest(10, "desviacion_porcentual")

print(
    mayores[
        [
            "promedio",
            "consumo_real",
            "desviacion_porcentual",
            "desc_clase_servicio",
            "periodo"
        ]
    ].to_string(index=False)
)

# 5. Diez anomalías con mayor desviación negativa
print("\n5. DIEZ MAYORES DESVIACIONES NEGATIVAS")

menores = df.nsmallest(10, "desviacion_porcentual")

print(
    menores[
        [
            "promedio",
            "consumo_real",
            "desviacion_porcentual",
            "desc_clase_servicio",
            "periodo"
        ]
    ].to_string(index=False)
)

print("\nANÁLISIS FINALIZADO.")
