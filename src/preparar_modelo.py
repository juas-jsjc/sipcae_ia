
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# 1. CARGAR DATOS
# ============================================================

RUTA_DATOS = "data/processed/consumos_limpios.csv"

df = pd.read_csv(RUTA_DATOS, sep="|")

print("=" * 65)
print("PREPARACIÓN DE DATOS PARA SIPCAE")
print("=" * 65)

# ============================================================
# 2. DEFINIR VARIABLES
# ============================================================

variables_predictoras = [
    "promedio",
    "dias_facturados",
    "factor",
    "estrato",
    "desc_clase_servicio",
    "periodo"
]

variable_objetivo = "consumo_real"
variable_grupo = "codigo"

# Indicadores que se conservan para control, no para entrenar
variables_control = [
    "consumo_ajustado_negativo",
    "consumo_imputado_faltante"
]

columnas_requeridas = (
    variables_predictoras
    + [variable_objetivo, variable_grupo]
    + variables_control
)

faltantes = [
    columna for columna in columnas_requeridas
    if columna not in df.columns
]

if faltantes:
    raise ValueError(f"Faltan columnas en el archivo: {faltantes}")

datos = df[columnas_requeridas].copy()

# ============================================================
# 3. PREPARAR VARIABLES
# ============================================================

variables_categoricas = [
    "estrato",
    "desc_clase_servicio",
    "periodo"
]

for columna in variables_categoricas:
    datos[columna] = datos[columna].astype(str)

# Verificar datos faltantes
if datos.isnull().any().any():
    print("\nValores faltantes encontrados:")
    print(datos.isnull().sum())
    raise ValueError("Revisar los valores faltantes antes de continuar.")

# ============================================================
# 4. SEPARAR ENTRENAMIENTO Y PRUEBA POR USUARIO
# ============================================================

X = datos[variables_predictoras]
y = datos[variable_objetivo]
grupos = datos[variable_grupo]

separador = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

indices_entrenamiento, indices_prueba = next(
    separador.split(X, y, groups=grupos)
)

# ============================================================
# 5. CONSTRUIR LOS CONJUNTOS
# ============================================================

datos_entrenamiento = X.iloc[indices_entrenamiento].copy()
datos_prueba = X.iloc[indices_prueba].copy()

datos_entrenamiento[variable_objetivo] = y.iloc[indices_entrenamiento]
datos_prueba[variable_objetivo] = y.iloc[indices_prueba]

for columna in variables_control:
    datos_entrenamiento[columna] = datos[columna].iloc[
        indices_entrenamiento
    ].values

    datos_prueba[columna] = datos[columna].iloc[
        indices_prueba
    ].values

# Conservar el código únicamente como identificador de grupo
datos_entrenamiento[variable_grupo] = grupos.iloc[
    indices_entrenamiento
].values

datos_prueba[variable_grupo] = grupos.iloc[
    indices_prueba
].values

# ============================================================
# 6. GUARDAR ARCHIVOS
# ============================================================

datos_entrenamiento.to_csv(
    "data/processed/entrenamiento.csv",
    index=False
)

datos_prueba.to_csv(
    "data/processed/prueba.csv",
    index=False
)

# ============================================================
# 7. VALIDAR SEPARACIÓN
# ============================================================

codigos_entrenamiento = grupos.iloc[indices_entrenamiento]
codigos_prueba = grupos.iloc[indices_prueba]

usuarios_compartidos = (
    set(codigos_entrenamiento)
    & set(codigos_prueba)
)

print("\nVARIABLES PREDICTORAS:")
for variable in variables_predictoras:
    print(f"- {variable}")

print("\nVARIABLES DE CONTROL:")
for variable in variables_control:
    print(f"- {variable}")

print(f"\nRegistros totales: {len(datos)}")
print(f"Registros de entrenamiento: {len(datos_entrenamiento)}")
print(f"Registros de prueba: {len(datos_prueba)}")

print(f"\nUsuarios entrenamiento: {codigos_entrenamiento.nunique()}")
print(f"Usuarios prueba: {codigos_prueba.nunique()}")
print(f"Usuarios compartidos: {len(usuarios_compartidos)}")

print("\nARCHIVOS GENERADOS:")
print("- data/processed/entrenamiento.csv")
print("- data/processed/prueba.csv")

print("\nPREPARACIÓN FINALIZADA.")