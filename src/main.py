#sipcae_ia

print("+-" * 30)
print("                       S I P C A E                       ")
print("+-" * 30)

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuración del estilo visual de Seaborn
sns.set_theme(style="whitegrid", context="talk")

print("=" * 60)
print("1. CARGA E INSPECCIÓN DEL DATASET")
print("=" * 60)

ruta_archivo = "consumos_104.csv"

try:
    df = pd.read_csv(
        ruta_archivo,
        sep="|",
        low_memory=False,
        encoding="cp1252"
        )
    print(
        f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas."
    )
except FileNotFoundError:
    print(
        f"Error: El archivo '{ruta_archivo}' no se encontró. Verifica la ruta."
    )
    # Creamos un DataFrame simulado para pruebas si el archivo no existe localmente
    df = pd.DataFrame()

if not df.empty:
    # Capturar métricas "Antes" para la tabla comparativa
    filas_antes = df.shape[0]
    columnas_antes = df.shape[1]
    nulos_antes = int(df.isnull().sum().sum())

    print("\n--- Primeras 5 filas ---")
    print(df.head())

    print("\n--- Información General ---")
    print(df.info())

    print("\n--- Valores Nulos por Columna ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    duplicados_iniciales = df.duplicated().sum()
    print(f"\nFilas duplicadas iniciales: {duplicados_iniciales}")

    # ============================================================
    # 2. TRATAMIENTO Y LIMPIEZA DE DATOS
    # ============================================================
    print("\n" + "=" * 60)
    print("2. TRATAMIENTO Y LIMPIEZA DE DATOS")
    print("=" * 60)

    df_limpio = df.copy()

    # A. Conversión de tipos de datos incorrectos a numéricos
    cols_numericas = [
        "consumo_real",
        "promedio",
        "carga",
        "Edad_Actual",
        "dias_facturados",
    ]
    for col in cols_numericas:
        if col in df_limpio.columns:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")

    # B. Tratamiento de valores nulos (imputación con la mediana en numéricas clave)
    if "consumo_real" in df_limpio.columns:
        med_consumo = df_limpio["consumo_real"].median()
        df_limpio["consumo_real"] = df_limpio["consumo_real"].fillna(
            med_consumo
        )

    if "promedio" in df_limpio.columns:
        med_promedio = df_limpio["promedio"].median()
        df_limpio["promedio"] = df_limpio["promedio"].fillna(med_promedio)

    if "carga" in df_limpio.columns:
        med_carga = df_limpio["carga"].median()
        df_limpio["carga"] = df_limpio["carga"].fillna(med_carga)

    if "Edad_Actual" in df_limpio.columns:
        med_edad = df_limpio["Edad_Actual"].median()
        df_limpio["Edad_Actual"] = df_limpio["Edad_Actual"].fillna(med_edad)

    # C. Eliminación de filas duplicadas
    df_limpio = df_limpio.drop_duplicates()

    # D. Tratamiento de valores atípicos o imposibles (ej: consumo o edad negativos)
    if "consumo_real" in df_limpio.columns:
        df_limpio = df_limpio[df_limpio["consumo_real"] >= 0]
    if "Edad_Actual" in df_limpio.columns:
        df_limpio = df_limpio[
            (df_limpio["Edad_Actual"] >= 0) & (df_limpio["Edad_Actual"] <= 120)
        ]

    # E. Exclusión de columnas irrelevantes para dejar un archivo limpio optimizado
    columnas_a_excluir = [
        "nombre",
        "direccion",
        "barrio",
        "codigo",
        "transformador",
        "trafo_id",
        "codigo_sig",
        "numero",
        "Consecutivo",
        "fecha_creacion",
        "Fecha_Ult_Instalacion",
        "Fecha_ultima_Rev",
        "fecha_lectura",
    ]
    cols_existentes = [
        col for col in columnas_a_excluir if col in df_limpio.columns
    ]
    df_limpio = df_limpio.drop(columns=cols_existentes)

    # F. Codificación de variables categóricas con pd.get_dummies
    if "desc_clase_servicio" in df_limpio.columns:
        df_limpio = pd.get_dummies(
            df_limpio, columns=["desc_clase_servicio"], prefix="clase_serv"
        )
        print("Codificación One-Hot aplicada a 'desc_clase_servicio'.")

    # Exportar dataset limpio
    os.makedirs("data", exist_ok=True)
    ruta_salida = "data/proyecto_limpio.csv"
    df_limpio.to_csv(ruta_salida, index=False, sep="|")
    print(f"Dataset limpio exportado exitosamente a '{ruta_salida}'")

    # Métricas "Después" para la tabla comparativa
    filas_despues = df_limpio.shape[0]
    columnas_despues = df_limpio.shape[1]
    nulos_despues = int(df_limpio.isnull().sum().sum())

    tabla_comparativa = pd.DataFrame(
        {
            "Métrica": ["Número de Filas", "Número de Columnas", "Valores Nulos"],
            "Antes": [filas_antes, columnas_antes, nulos_antes],
            "Después": [filas_despues, columnas_despues, nulos_despues],
        }
    )
    print("\n--- TABLA ANTES / DESPUÉS ---")
    print(tabla_comparativa.to_string(index=False))

    # ============================================================
    # 3. VISUALIZACIÓN CON SEABORN
    # ============================================================
    print("\n" + "=" * 60)
    print("3. VISUALIZACIÓN CON SEABORN")
    print("=" * 60)

    # Gráfico 1: Relación entre variables de entrada y la variable objetivo (Promedio vs Consumo Real)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df_limpio,
        x="promedio",
        y="consumo_real",
        hue="estrato" if "estrato" in df_limpio.columns else None,
        palette="viridis",
        alpha=0.7,
    )
    plt.title("Relación: Consumo Promedio vs Consumo Real")
    plt.xlabel("Consumo Promedio")
    plt.ylabel("Consumo Real")
    plt.tight_layout()
    plt.savefig("relacion_promedio_consumo.png", dpi=150)
    #plt.show()
    plt.savefig("grafico_1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(
        "Interpretación Gráfico 1: Muestra una relación lineal positiva sólida entre el consumo promedio y el consumo real. Apoya la decisión de utilizar el promedio histórico como variable predictora clave en los modelos analíticos."
    )

    # Gráfico 2: Distribución de la variable objetivo (consumo_real)
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df_limpio["consumo_real"], bins=30, kde=True, color="teal"
    )
    plt.title("Distribución del Consumo Real")
    plt.xlabel("Consumo Real")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig("distribucion_consumo_real.png", dpi=150)
    #plt.show()
    plt.savefig("grafico_2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(
        "Interpretación Gráfico 2: Evidencia un sesgo positivo pronunciado con alta concentración en consumos bajos y presencia de valores altos. Justifica el uso de la mediana como umbral objetivo para evitar sesgos por valores extremos."
    )

    # Gráfico 3: Distribución de Carga por Estrato
    if "carga" in df_limpio.columns and "estrato" in df_limpio.columns:
        plt.figure(figsize=(9, 5))
        sns.boxplot(
            data=df_limpio,
            x="estrato",
            y="carga",
            hue="estrato",
            palette="Set2",
            legend=False            
        )
        plt.title("Distribución de la Carga Instalada por Estrato")
        plt.xlabel("Estrato")
        plt.ylabel("Carga Instalada")
        plt.tight_layout()
        plt.savefig("carga_por_estrato.png", dpi=150)
        #plt.show()
        plt.savefig("grafico_3.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(
            "Interpretación Gráfico 3: Ilustra cómo la carga instalada tiende a incrementarse conforme aumenta el estrato socioeconómico. Valida la segmentación de la demanda eléctrica según el perfil socioeconómico del usuario."
        )

    # ============================================================
    # 4. PRIMER MODELO: PERCEPTRÓN DESDE CERO (CON NUMPY)
    # ============================================================
    print("\n" + "=" * 60)
    print("4. PRIMER MODELO: PERCEPTRÓN DESDE CERO (NUMPY)")
    print("=" * 60)

    # Definición de la variable objetivo binaria (0 / 1) a partir de 'consumo_real'
    umbral = df_limpio["consumo_real"].median()
    y = (df_limpio["consumo_real"] > umbral).astype(int).to_numpy()

    print(
        f"Definición de Objetivo: Se binarizó 'consumo_real' usando la mediana ({umbral:.2f}). "
        "La columna original 'consumo_real' queda excluida de las entradas para evitar fuga de datos (data leakage)."
    )

    # Selección de dos variables de entrada numéricas ('promedio' y 'carga')
    X_raw = df_limpio[["promedio", "carga"]].to_numpy()

    # Estandarización de las entradas (Media 0, Desviación 1)
    media_x = X_raw.mean(axis=0)
    desv_x = X_raw.std(axis=0)
    desv_x = np.where(desv_x == 0, 1.0, desv_x)  # Evitar división por cero
    X = (X_raw.astype(float) - media_x) / desv_x

    # Funciones del Perceptrón
    def suma_ponderada(X_in, w_in, b_in):
        return X_in @ w_in + b_in

    def sigmoide(z):
        z_clip = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clip))

    def predecir_probabilidad(X_in, w_in, b_in):
        return sigmoide(suma_ponderada(X_in, w_in, b_in))

    def calcular_perdida(y_real, y_prob):
        epsilon = 1e-15
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        return -np.mean(
            y_real * np.log(y_prob) + (1 - y_real) * np.log(1 - y_prob)
        )

    def calcular_exactitud(y_real, y_prob):
        return np.mean((y_prob >= 0.5).astype(int) == y_real)

    def calcular_gradientes(X_in, y_real, y_prob):
        n = len(y_real)
        error = y_prob - y_real
        grad_w = (X_in.T @ error) / n
        grad_b = np.mean(error)
        return grad_w, grad_b

    # Inicialización de parámetros
    np.random.seed(42)
    n_caracteristicas = X.shape[1]
    w = np.zeros(n_caracteristicas)
    b = 0.0
    tasa_aprendizaje = 0.1
    epocas = 500

    historial_perdida = []
    historial_exactitud = []

    # Entrenamiento por Descenso del Gradiente
    for epoca in range(1, epocas + 1):
        y_prob = predecir_probabilidad(X, w, b)
        grad_w, grad_b = calcular_gradientes(X, y, y_prob)

        w -= tasa_aprendizaje * grad_w
        b -= tasa_aprendizaje * grad_b

        perdida_actual = calcular_perdida(y, y_prob)
        exactitud_actual = calcular_exactitud(y, y_prob)

        historial_perdida.append(perdida_actual)
        historial_exactitud.append(exactitud_actual)

    # Resultados Finales
    y_prob_final = predecir_probabilidad(X, w, b)
    exactitud_final = calcular_exactitud(y, y_prob_final)
    perdida_final = calcular_perdida(y, y_prob_final)

    print("\n--- RESULTADOS DEL PERCEPTRÓN ---")
    print(f"Pesos finales (w1, w2): [{w[0]:.4f}, {w[1]:.4f}]")
    print(f"Sesgo final (b): {b:.4f}")
    print(f"Pérdida inicial: {historial_perdida[0]:.4f}")
    print(f"Pérdida final: {perdida_final:.4f}")
    print(f"Exactitud (Accuracy): {exactitud_final * 100:.2f}%")

    # Gráfico de Curva de Aprendizaje
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(historial_perdida, color="red")
    ax1.set_title("Pérdida por Época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida Logarítmica")
    ax1.grid(True, alpha=0.3)

    ax2.plot(historial_exactitud, color="green")
    ax2.set_title("Exactitud por Época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Exactitud")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("curva_aprendizaje_perceptron.png", dpi=150)
    plt.close()

    # Gráfico de Frontera de Decisión (en escala original)
    x1_orig = np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), 100)
    x1_esc = (x1_orig - media_x[0]) / desv_x[0]
    # Despeje de x2 en la frontera z = 0 -> w1*x1_esc + w2*x2_esc + b = 0
    x2_esc = -(w[0] * x1_esc + b) / w[1]
    x2_orig = x2_esc * desv_x[1] + media_x[1]

    plt.figure(figsize=(9, 6))
    plt.scatter(
        X_raw[y == 1, 0],
        X_raw[y == 1, 1],
        color="green",
        edgecolors="black",
        label="Consumo Alto (1)",
        alpha=0.7,
    )
    plt.scatter(
        X_raw[y == 0, 0],
        X_raw[y == 0, 1],
        color="orange",
        marker="s",
        edgecolors="black",
        label="Consumo Bajo (0)",
        alpha=0.7,
    )
    plt.plot(
        x1_orig,
        x2_orig,
        color="blue",
        linewidth=2.5,
        label="Frontera de Decisión",
    )
    plt.title("Frontera de Decisión del Perceptrón")
    plt.xlabel("Consumo Promedio")
    plt.ylabel("Carga Instalada")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("frontera_decision_perceptron.png", dpi=150)
    #plt.show()
    plt.savefig(
        "perceptron_resultado.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(
        "\nInterpretación del Modelo:\n"
        "- Signo de los pesos: Pesos positivos indican que un mayor consumo promedio y mayor carga incrementan la probabilidad de pertenecer a la clase 1 (consumo alto).\n"
        "- Separabilidad lineal: Los datos muestran una separación lineal moderada, aunque existen zonas de solapamiento donde los clientes comparten métricas similares.\n"
        "- Limitaciones: El perceptrón simple utiliza una única línea de decisión recta, lo que limita su desempeño ante relaciones complejas o no lineales presentes en el comportamiento real de los usuarios."
    )