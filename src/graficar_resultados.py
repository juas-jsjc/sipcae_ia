
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

RUTA_RESULTADOS = (
    "data/processed/resultados_validacion_cruzada.csv"
)

CARPETA_GRAFICOS = "reports/figures"

os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

# Cargar resultados de validación cruzada
datos = pd.read_csv(RUTA_RESULTADOS)

# ============================================================
# 2. CONFIGURACIÓN VISUAL
# ============================================================

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

# Orden de presentación
orden_modelos = [
    "Promedio histórico",
    "Random Forest",
    "Gradient Boosting"
]

# ============================================================
# 3. GRÁFICO DE COMPARACIÓN GENERAL
# ============================================================

resumen = (
    datos.groupby("Modelo")[["MAE", "RMSE", "R2"]]
    .mean()
    .reindex(orden_modelos)
)

fig, ejes = plt.subplots(1, 3, figsize=(16, 5))

metricas = [
    ("MAE", "Error absoluto medio"),
    ("RMSE", "Raíz del error cuadrático medio"),
    ("R2", "Coeficiente de determinación")
]

for eje, (metrica, titulo) in zip(ejes, metricas):

    resumen[metrica].plot(
        kind="bar",
        ax=eje,
        rot=25
    )

    eje.set_title(titulo)
    eje.set_xlabel("")
    eje.set_ylabel(metrica)
    eje.grid(axis="y", alpha=0.3)

    for barra in eje.patches:
        valor = barra.get_height()

        eje.annotate(
            f"{valor:.3f}",
            (barra.get_x() + barra.get_width() / 2, valor),
            ha="center",
            va="bottom",
            fontsize=8
        )

fig.suptitle(
    "Comparación general de modelos - SIPCAE",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    f"{CARPETA_GRAFICOS}/06_comparacion_modelos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 4. GRÁFICO DE MAE POR PARTICIÓN
# ============================================================

plt.figure(figsize=(10, 6))

for modelo in orden_modelos:

    subconjunto = datos[datos["Modelo"] == modelo]

    plt.plot(
        subconjunto["Particion"],
        subconjunto["MAE"],
        marker="o",
        linewidth=2,
        label=modelo
    )

plt.title("MAE por partición - Validación cruzada")
plt.xlabel("Partición")
plt.ylabel("MAE")
plt.xticks([1, 2, 3, 4, 5])
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{CARPETA_GRAFICOS}/07_mae_por_particion.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 5. GRÁFICO DE R2 POR PARTICIÓN
# ============================================================

plt.figure(figsize=(10, 6))

for modelo in orden_modelos:

    subconjunto = datos[datos["Modelo"] == modelo]

    plt.plot(
        subconjunto["Particion"],
        subconjunto["R2"],
        marker="o",
        linewidth=2,
        label=modelo
    )

plt.title("R² por partición - Validación cruzada")
plt.xlabel("Partición")
plt.ylabel("R²")
plt.xticks([1, 2, 3, 4, 5])
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{CARPETA_GRAFICOS}/08_r2_por_particion.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 6. CONFIRMAR RESULTADOS
# ============================================================

print("=" * 60)
print("GRÁFICOS DE EVALUACIÓN GENERADOS - SIPCAE")
print("=" * 60)

print("\nArchivos generados:")

print("- 06_comparacion_modelos.png")
print("- 07_mae_por_particion.png")
print("- 08_r2_por_particion.png")

print("\nResumen de métricas:")
print(resumen.round(4))

print("\nProceso finalizado correctamente.")