
import pandas as pd

ruta = "data/processed/consumos_limpios.csv"

df = pd.read_csv(
    ruta,
    sep="|",
    encoding="utf-8-sig",
    low_memory=False
)

print("=" * 65)
print("CARACTERIZACIÓN DE CATEGORÍAS - SIPCAE")
print("=" * 65)

# 1. Estratos
print("\n1. ANÁLISIS POR ESTRATO")

estratos = df.groupby("estrato").agg(
    registros=("consumo_real", "size"),
    consumos_cero=("consumo_real", lambda x: (x == 0).sum()),
    mediana_general=("consumo_real", "median"),
    mediana_positivos=(
        "consumo_real",
        lambda x: x[x > 0].median()
    ),
    promedio_positivos=(
        "consumo_real",
        lambda x: x[x > 0].mean()
    )
)

estratos["porcentaje_cero"] = (
    estratos["consumos_cero"] /
    estratos["registros"] * 100
).round(2)

print(estratos.round(2).to_string())

# 2. Clases de servicio
print("\n2. ANÁLISIS POR CLASE DE SERVICIO")

clases = df.groupby("desc_clase_servicio").agg(
    registros=("consumo_real", "size"),
    consumos_cero=("consumo_real", lambda x: (x == 0).sum()),
    mediana_general=("consumo_real", "median"),
    mediana_positivos=(
        "consumo_real",
        lambda x: x[x > 0].median()
    ),
    promedio_positivos=(
        "consumo_real",
        lambda x: x[x > 0].mean()
    )
)

clases["porcentaje_cero"] = (
    clases["consumos_cero"] /
    clases["registros"] * 100
).round(2)

print(clases.round(2).to_string())

# 3. Estrato por clase de servicio
print("\n3. DISTRIBUCIÓN DE ESTRATOS POR CLASE")

print(
    pd.crosstab(
        df["desc_clase_servicio"],
        df["estrato"]
    ).to_string()
)

print("\nANÁLISIS FINALIZADO")