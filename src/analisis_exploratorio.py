
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración
ruta = "data/processed/consumos_limpios.csv"
carpeta_salida = Path("reports/figures")
carpeta_salida.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")

# Cargar datos
df = pd.read_csv(
    ruta,
    sep="|",
    encoding="utf-8-sig",
    low_memory=False
)

# Variables numéricas
for columna in [
    "consumo_real", "promedio", "estrato",
    "dias_facturados", "factor"
]:
    df[columna] = pd.to_numeric(df[columna], errors="coerce")

# Excluir registros con consumo cero de algunos análisis
# para distinguir la distribución de consumos positivos.
df_positivos = df[df["consumo_real"] > 0].copy()

print("=" * 60)
print("ANÁLISIS EXPLORATORIO - SIPCAE")
print("=" * 60)

# --------------------------------------------------
# 1. DISTRIBUCIÓN DEL CONSUMO REAL
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df_positivos,
    x="consumo_real",
    bins=50,
    kde=True
)

plt.title("Distribución del consumo real positivo")
plt.xlabel("Consumo real")
plt.ylabel("Frecuencia")
plt.tight_layout()

plt.savefig(
    carpeta_salida / "01_distribucion_consumo.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 2. CONSUMO POR ESTRATO
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df_positivos,
    x="estrato",
    y="consumo_real"
)

plt.title("Distribución del consumo por estrato")
plt.xlabel("Estrato")
plt.ylabel("Consumo real")
plt.tight_layout()

plt.savefig(
    carpeta_salida / "02_consumo_por_estrato.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 3. EVOLUCIÓN DEL CONSUMO POR PERIODO
# --------------------------------------------------

orden_periodos = ["septiembre", "octubre", "noviembre"]

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="periodo",
    y="consumo_real",
    order=orden_periodos
)

plt.title("Evolución del consumo real por periodo")
plt.xlabel("Periodo")
plt.ylabel("Consumo real")
plt.tight_layout()

plt.savefig(
    carpeta_salida / "03_evolucion_temporal.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 4. CONSUMO REAL FRENTE AL PROMEDIO HISTÓRICO
# --------------------------------------------------

muestra = df.sample(
    n=min(2000, len(df)),
    random_state=42
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=muestra,
    x="promedio",
    y="consumo_real",
    hue="periodo",
    alpha=0.5
)

plt.title("Consumo real frente al promedio histórico")
plt.xlabel("Promedio histórico")
plt.ylabel("Consumo real")
plt.tight_layout()

plt.savefig(
    carpeta_salida / "04_consumo_vs_promedio.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 5. CONSUMO POR CLASE DE SERVICIO
# --------------------------------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df_positivos,
    x="desc_clase_servicio",
    y="consumo_real"
)

plt.title("Consumo real por clase de servicio")
plt.xlabel("Clase de servicio")
plt.ylabel("Consumo real")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    carpeta_salida / "05_consumo_por_clase.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# RESUMEN ESTADÍSTICO
# --------------------------------------------------

print("\nMEDIANA DEL CONSUMO POR PERIODO")

print(
    df.groupby("periodo")["consumo_real"]
    .median()
)

print("\nMEDIANA DEL CONSUMO POR ESTRATO")

print(
    df.groupby("estrato")["consumo_real"]
    .median()
)

print("\nMEDIANA DEL CONSUMO POR CLASE DE SERVICIO")

print(
    df.groupby("desc_clase_servicio")["consumo_real"]
    .median()
)

print("\nGRÁFICOS GENERADOS:")

for archivo in sorted(carpeta_salida.glob("*.png")):
    print(archivo)

print("\nANÁLISIS EXPLORATORIO FINALIZADO")