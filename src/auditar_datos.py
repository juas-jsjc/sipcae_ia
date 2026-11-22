# Auditar Datos

import pandas as pd

# Cargar datos consolidados
ruta = "data/processed/consumos_consolidados.csv"

df = pd.read_csv(
    ruta,
    sep="|",
    encoding="utf-8-sig",
    low_memory=False
)

print("=" * 60)
print("AUDITORÍA DE CALIDAD DE DATOS - SIPCAE")
print("=" * 60)

# 1. Información general
print("\n1. INFORMACIÓN GENERAL")
print("Dimensiones:", df.shape)
print("Columnas:", len(df.columns))
print("Códigos únicos:", df["codigo"].nunique())
print("Periodos:", df["periodo"].unique())

# 2. Valores faltantes
print("\n2. VALORES FALTANTES")

faltantes = pd.DataFrame({
    "cantidad": df.isnull().sum(),
    "porcentaje": (df.isnull().mean() * 100).round(2)
})

faltantes = faltantes[faltantes["cantidad"] > 0]
print(faltantes.sort_values("cantidad", ascending=False))

# 3. Registros duplicados
print("\n3. REGISTROS DUPLICADOS")

print("Filas completamente duplicadas:", df.duplicated().sum())

duplicados_codigo = df.duplicated(
    subset=["codigo", "periodo"]
).sum()

print("Códigos duplicados dentro de un periodo:", duplicados_codigo)

# 4. Consumos negativos
print("\n4. CONSUMOS NEGATIVOS")

df["consumo_real"] = pd.to_numeric(
    df["consumo_real"], errors="coerce"
)

negativos = df[df["consumo_real"] < 0]

print("Total de consumos negativos:", len(negativos))

print("\nNegativos por periodo:")
print(
    negativos.groupby("periodo")["consumo_real"]
    .agg(["count", "min", "max", "mean"])
)

# 5. Estadísticas del consumo
print("\n5. ESTADÍSTICAS DE CONSUMO REAL")

print(
    df.groupby("periodo")["consumo_real"]
    .describe()
    .round(2)
)

# 6. Valores extremos mediante percentiles
print("\n6. PERCENTILES DEL CONSUMO")

print(
    df.groupby("periodo")["consumo_real"]
    .quantile([0.01, 0.05, 0.50, 0.95, 0.99])
    .unstack()
    .round(2)
)

# 7. Consistencia temporal
print("\n7. CONSISTENCIA TEMPORAL")

registros_codigo = df.groupby("codigo")["periodo"].nunique()

print("Códigos con tres periodos:", (registros_codigo == 3).sum())
print("Códigos con menos de tres periodos:", (registros_codigo < 3).sum())

print("\nRegistros por periodo:")
print(df["periodo"].value_counts())

print("\nAUDITORÍA FINALIZADA")