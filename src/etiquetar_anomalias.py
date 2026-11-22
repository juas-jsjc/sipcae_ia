
import pandas as pd

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

RUTA_ENTRENAMIENTO = "data/processed/entrenamiento.csv"
RUTA_PRUEBA = "data/processed/prueba.csv"

UMBRAL_INFERIOR = 0.50
UMBRAL_SUPERIOR = 1.50

# ============================================================
# 2. FUNCIÓN PARA CLASIFICAR ANOMALÍAS
# ============================================================

def etiquetar_datos(df):
    datos = df.copy()

    # Excluir registros sin referencia histórica válida
    # y consumos ajustados o imputados.
    validos = (
        (datos["promedio"] > 0)
        & (datos["consumo_ajustado_negativo"] == 0)
        & (datos["consumo_imputado_faltante"] == 0)
    )

    datos = datos.loc[validos].copy()

    # Calcular límites individuales según el promedio histórico
    limite_inferior = datos["promedio"] * UMBRAL_INFERIOR
    limite_superior = datos["promedio"] * UMBRAL_SUPERIOR

    # Crear etiqueta:
    # 1 = anómalo
    # 0 = normal
    datos["anomalia"] = (
        (datos["consumo_real"] < limite_inferior)
        | (datos["consumo_real"] > limite_superior)
    ).astype(int)

    return datos


# ============================================================
# 3. CARGAR CONJUNTOS
# ============================================================

entrenamiento = pd.read_csv(RUTA_ENTRENAMIENTO)
prueba = pd.read_csv(RUTA_PRUEBA)

print("=" * 65)
print("CLASIFICACIÓN DE ANOMALÍAS - SIPCAE")
print("=" * 65)

# ============================================================
# 4. GENERAR ETIQUETAS
# ============================================================

entrenamiento_etiquetado = etiquetar_datos(entrenamiento)
prueba_etiquetada = etiquetar_datos(prueba)

# ============================================================
# 5. GUARDAR RESULTADOS
# ============================================================

entrenamiento_etiquetado.to_csv(
    "data/processed/entrenamiento_etiquetado.csv",
    index=False
)

prueba_etiquetada.to_csv(
    "data/processed/prueba_etiquetada.csv",
    index=False
)

# ============================================================
# 6. MOSTRAR RESULTADOS
# ============================================================

def mostrar_resumen(nombre, datos):
    print(f"\n{nombre}")
    print("-" * 40)
    print(f"Registros clasificados: {len(datos)}")

    print("\nDistribución de clases:")
    print(datos["anomalia"].value_counts().sort_index())

    print("\nPorcentajes:")
    print(
        (datos["anomalia"].value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )

mostrar_resumen("ENTRENAMIENTO", entrenamiento_etiquetado)
mostrar_resumen("PRUEBA", prueba_etiquetada)

print("\nTOTAL DE REGISTROS CLASIFICADOS:")
print(
    len(entrenamiento_etiquetado)
    + len(prueba_etiquetada)
)

print("\nARCHIVOS GENERADOS:")
print("- data/processed/entrenamiento_etiquetado.csv")
print("- data/processed/prueba_etiquetada.csv")

print("\nCLASIFICACIÓN FINALIZADA.")
