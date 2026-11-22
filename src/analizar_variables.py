
import pandas as pd

ruta = "data/processed/consumos_limpios.csv"

df = pd.read_csv(
    ruta,
    sep="|",
    encoding="utf-8-sig",
    low_memory=False
)

variables = [
    "consumo_real",
    "promedio",
    "dias_facturados",
    "factor",
    "carga",
    "carga inst",
    "Edad_Actual",
    "%desviacion"
]

print("=" * 70)
print("DIAGNÓSTICO DE VARIABLES PREDICTORAS - SIPCAE")
print("=" * 70)

# Convertir variables a formato numérico
for columna in variables:
    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )

# Estadísticas descriptivas
print("\n1. ESTADÍSTICAS DESCRIPTIVAS")

print(
    df[variables]
    .describe()
    .round(2)
    .to_string()
)

# Valores faltantes
print("\n2. VALORES FALTANTES")

faltantes = pd.DataFrame({
    "cantidad": df[variables].isna().sum(),
    "porcentaje": (
        df[variables].isna().mean() * 100
    ).round(2)
})

print(faltantes.to_string())

# Valores iguales a cero
print("\n3. VALORES IGUALES A CERO")

print(
    (df[variables] == 0)
    .sum()
    .to_string()
)

# Correlaciones de Spearman
print("\n4. CORRELACIONES DE SPEARMAN")

correlaciones = df[variables].corr(
    method="spearman"
)

print(correlaciones.round(3).to_string())

print("\nDIAGNÓSTICO FINALIZADO")
