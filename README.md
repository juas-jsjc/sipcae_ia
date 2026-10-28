# ⚡ SISTEMA INTELIGENTE DE PREDICCIÓN Y CONTROL DE ANOMALÍAS ENERGÉTICAS (SIPCAE) 

Repositorio oficial para el desarrollo del proyecto de Inteligencia Artificial. Este sistema implementa un pipeline completo de Ciencia de Datos: desde la preparación y limpieza de datos estructurados con Pandas, pasando por el análisis visual con Seaborn, hasta la implementación manual de un modelo de clasificación (Perceptrón) desde cero utilizando NumPy dentro de un entorno contenedorizado con Docker.

---

## 👥 Integrantes del Grupo
* **Integrante:** Juan Sebastián Jiménez Cardona
* **Docente:** Jhon James Cano Sánchez

---

## 🚀 Avance 2: Resultados y Ejecución

La sección del **Avance 2** comprende tres fases fundamentales del proyecto integrador:

### 1. Resumen de Resultados del Avance 2
* **Limpieza de Datos:** Se procesó un archivo plano masivo (con separador pipe `|`), eliminando nulos mediante imputación con la mediana, descartando duplicados y excluyendo columnas irrelevantes para evitar ruido en el modelo. El dataset resultante se exportó limpiamente a `data/proyecto_limpio.csv`.
* **Análisis Visual (Seaborn):** Se generaron gráficos clave que relacionan el consumo promedio histórico con el consumo real (evidenciando una fuerte correlación lineal positiva), la distribución de la variable objetivo y el comportamiento de la carga instalada por estrato socioeconómico.
* **Modelo Perceptrón desde Cero (NumPy):** Se implementó un perceptrón binarizando la variable objetivo `consumo_real` (usando la mediana como umbral) con dos variables de entrada estandarizadas (`promedio` y `carga`). Tras 500 épocas de entrenamiento por descenso del gradiente, el modelo alcanzó una **exactitud superior al 85%**, validada mediante su respectiva curva de aprendizaje y frontera de decisión.

---

## 🐳 Instrucciones de Ejecución con Docker

Para garantizar la reproducibilidad del entorno y evitar conflictos de librerías, todo el código se ejecuta dentro de un contenedor Docker. Sigue los pasos a continuación:

### Prerrequisitos
* Tener instalado [Docker](https://www.docs.docker.com/get-docker/) en tu sistema.

### Paso 1: Clonar el Repositorio
Abre tu terminal y clona este repositorio en tu máquina local:
```bash
git clone https://github.com/juas-jsjc/sipcae_ia
cd sipcae_ia/src
