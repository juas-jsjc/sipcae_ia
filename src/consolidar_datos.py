#Consolidar datos


import pandas as pd
from pathlib import Path

# Rutas de los archivos originales
archivos = {
    "septiembre": "data/raw/consumos_104_septiembre.csv",
    "octubre": "data/raw/consumos_104_octubre.csv",
    "noviembre": "data/raw/consumos_104_noviembre.csv"
}

# Cargar los tres periodos
datos = {}

for periodo, ruta in archivos.items():
    df = pd.read_csv(
        ruta,
        sep="|",
        encoding="cp1252",
        low_memory=False
    )

    datos[periodo] = df
    print(f"{periodo.capitalize()}: {len(df)} registros")

# Verificar que los archivos tengan las mismas columnas
columnas_base = list(datos["septiembre"].columns)

for periodo, df in datos.items():
    if list(df.columns) != columnas_base:
        raise ValueError(
            f"Las columnas de {periodo} no coinciden con septiembre."
        )

# Obtener los códigos comunes a los tres periodos
conjuntos_codigos = [
    set(df["codigo"].dropna().astype(str))
    for df in datos.values()
]

codigos_comunes = set.intersection(*conjuntos_codigos)

print(f"\nCódigos comunes: {len(codigos_comunes)}")

# Filtrar y agregar la columna periodo
consolidados = []

for periodo, df in datos.items():
    df = df.copy()
    df["codigo"] = df["codigo"].astype(str)

    df = df[df["codigo"].isin(codigos_comunes)].copy()
    df["periodo"] = periodo

    consolidados.append(df)

# Unir los tres periodos
df_final = pd.concat(consolidados, ignore_index=True)

# Guardar el archivo consolidado
ruta_salida = Path("data/processed/consumos_consolidados.csv")
ruta_salida.parent.mkdir(parents=True, exist_ok=True)

df_final.to_csv(
    ruta_salida,
    sep="|",
    index=False,
    encoding="utf-8-sig"
)

# Mostrar resultados
print("\nCONSOLIDACIÓN FINAL")
print(f"Total de registros: {len(df_final)}")
print(f"Códigos únicos: {df_final['codigo'].nunique()}")
print(f"Periodos: {df_final['periodo'].nunique()}")
print(f"Registros esperados: {len(codigos_comunes) * 3}")
print(f"Archivo generado: {ruta_salida}")

print("\nRegistros por periodo:")
print(df_final["periodo"].value_counts())