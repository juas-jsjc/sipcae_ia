
import joblib
import pandas as pd

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

RUTA_MODELO = "models/mejor_modelo_sipcae.joblib"

UMBRAL_INFERIOR = 0.50
UMBRAL_SUPERIOR = 1.50

# ============================================================
# 2. CARGAR MODELO
# ============================================================

print("=" * 65)
print("SIPCAE - PREDICCIÓN Y DETECCIÓN DE ANOMALÍAS")
print("=" * 65)

try:
    modelo = joblib.load(RUTA_MODELO)
    print("\nModelo Gradient Boosting cargado correctamente.")

except FileNotFoundError:
    print("\nERROR: No se encontró el modelo guardado.")
    raise SystemExit(1)

# ============================================================
# 3. INGRESAR INFORMACIÓN DEL REGISTRO
# ============================================================

def solicitar_numero(mensaje, minimo=None, mayor_que_minimo=False):

    while True:
        try:
            valor = float(input(mensaje))

            if minimo is not None:
                if mayor_que_minimo and valor <= minimo:
                    print(f"El valor debe ser mayor que {minimo}.")
                    continue

                if not mayor_que_minimo and valor < minimo:
                    print(f"El valor no puede ser menor que {minimo}.")
                    continue

            return valor

        except ValueError:
            print("Ingrese un valor numérico válido.")


def solicitar_opcion(mensaje, opciones):

    print(f"\n{mensaje}")

    for numero, opcion in enumerate(opciones, start=1):
        print(f"{numero}. {opcion}")

    while True:
        try:
            seleccion = int(input("Seleccione una opción: "))

            if 1 <= seleccion <= len(opciones):
                return opciones[seleccion - 1]

            print("Seleccione una opción de la lista.")

        except ValueError:
            print("Ingrese un número válido.")


print("\nINGRESE LOS DATOS DEL REGISTRO")
print("-" * 45)

promedio = solicitar_numero(
    "Promedio histórico de consumo (kWh): ",
    minimo=0,
    mayor_que_minimo=True
)

dias_facturados = solicitar_numero(
    "Días facturados: ",
    minimo=1
)

factor = solicitar_numero(
    "Factor de medida: ",
    minimo=0,
    mayor_que_minimo=True
)

estrato = solicitar_opcion(
    "Estrato:",
    ["0", "1", "2", "3", "4", "5", "6"]
)

clase_servicio = solicitar_opcion(
    "Clase de servicio:",
    [
        "Areas_Comunes",
        "Comercial",
        "Industrial",
        "Oficial",
        "Provisionales",
        "Residencial"
    ]
)

periodo = solicitar_opcion(
    "Periodo:",
    ["septiembre", "octubre", "noviembre"]
)

consumo_real = solicitar_numero(
    "\nConsumo real registrado (kWh): ",
    minimo=0
)

# ============================================================
# 4. PREPARAR DATOS PARA EL MODELO
# ============================================================

nuevo_registro = pd.DataFrame([{
    "promedio": promedio,
    "dias_facturados": dias_facturados,
    "factor": factor,
    "estrato": estrato,
    "desc_clase_servicio": clase_servicio,
    "periodo": periodo
}])

# ============================================================
# 5. PREDECIR CON GRADIENT BOOSTING
# ============================================================

consumo_estimado = modelo.predict(nuevo_registro)[0]

# Evitar estimaciones negativas
consumo_estimado = max(consumo_estimado, 0)

# ============================================================
# 6. CALCULAR DESVIACIÓN HISTÓRICA
# ============================================================

desviacion = (
    (consumo_real - promedio) / promedio
) * 100

limite_inferior = promedio * UMBRAL_INFERIOR
limite_superior = promedio * UMBRAL_SUPERIOR

# ============================================================
# 7. CLASIFICAR COMPORTAMIENTO
# ============================================================

if consumo_real < limite_inferior:
    resultado = "ANOMALIA BAJA"

elif consumo_real > limite_superior:
    resultado = "ANOMALIA ALTA"

else:
    resultado = "NORMAL"

# ============================================================
# 8. MOSTRAR RESULTADOS
# ============================================================

print("\n" + "=" * 65)
print("RESULTADOS DEL ANÁLISIS SIPCAE")
print("=" * 65)

print("\nINFORMACIÓN DEL REGISTRO")
print("-" * 45)
print(f"Promedio histórico:       {promedio:.2f} kWh")
print(f"Consumo real:             {consumo_real:.2f} kWh")
print(f"Consumo estimado ML:      {consumo_estimado:.2f} kWh")

print("\nANÁLISIS DE DESVIACIÓN")
print("-" * 45)
print(f"Límite inferior:          {limite_inferior:.2f} kWh")
print(f"Límite superior:          {limite_superior:.2f} kWh")
print(f"Desviación porcentual:    {desviacion:.2f}%")

print("\nCLASIFICACIÓN")
print("-" * 45)
print(f"Resultado: {resultado}")

if resultado == "NORMAL":
    print("El consumo se encuentra dentro del rango esperado.")

elif resultado == "ANOMALIA BAJA":
    print("El consumo está por debajo del límite histórico.")
    print("Se recomienda verificar las condiciones del registro.")

else:
    print("El consumo supera el límite histórico.")
    print("Se recomienda revisar el comportamiento registrado.")

print("\nNOTA:")
print(
    "Una anomalía estadística es una señal para revisión; "
    "no constituye por sí sola evidencia de fraude o falla."
)

print("\nANÁLISIS FINALIZADO.")
