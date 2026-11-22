
import pandas as pd
from pathlib import Path

# Rutas
ruta_entrada = "data/processed/consumos_consolidados.csv"
ruta_salida = "data/processed/consumos_limpios.csv"

# Cargar datos
df = pd.read_csv(
    ruta_entrada,
    sep="|",
    encoding="utf-8-sig",
    low_memory=False
)

print("=" * 60)
print("LIMPIEZA DE DATOS - SIPCAE")
print("=" * 60)

# Convertir consumo_real a formato numérico
df["consumo_real"] = pd.to_numeric(
    df["consumo_real"],
    errors="coerce"
)

# Identificar valores antes de la transformación
negativos = (df["consumo_real"] < 0).sum()
faltantes = df["consumo_real"].isna().sum()

# Crear indicadores de los valores transformados
df["consumo_ajustado_negativo"] = (
    df["consumo_real"] < 0
).astype(int)

df["consumo_imputado_faltante"] = (
    df["consumo_real"].isna()
).astype(int)

print(f"\nConsumos negativos: {negativos}")
print(f"Consumos faltantes: {faltantes}")

# Reemplazar negativos y faltantes por cero
df["consumo_real"] = df["consumo_real"].fillna(0)
df.loc[df["consumo_real"] < 0, "consumo_real"] = 0

print("\nTRANSFORMACIÓN APLICADA")
print(f"Negativos reemplazados: {negativos}")
print(f"Faltantes reemplazados: {faltantes}")

# Verificar resultados
print("\nVERIFICACIÓN")

print(
    "Consumos negativos restantes:",
    (df["consumo_real"] < 0).sum()
)

print(
    "Consumos faltantes restantes:",
    df["consumo_real"].isna().sum()
)

print(
    "Consumos iguales a cero:",
    (df["consumo_real"] == 0).sum()
)

# Estadísticas por periodo
print("\nESTADÍSTICAS POR PERIODO")

print(
    df.groupby("periodo")["consumo_real"]
    .describe()
    .round(2)
)

# Guardar archivo procesado
Path(ruta_salida).parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    ruta_salida,
    sep="|",
    index=False,
    encoding="utf-8-sig"
)

print("\nLIMPIEZA FINALIZADA")
print("Registros:", len(df))
print("Códigos únicos:", df["codigo"].nunique())
print("Valores faltantes:", df["consumo_real"].isna().sum())
print("Valores negativos:", (df["consumo_real"] < 0).sum())
print("Archivo generado:", ruta_salida)