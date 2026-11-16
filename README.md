# Clasificación de Calidad del Vino (EDA y Machine Learning)

Proyecto modularizado en Python para el Análisis Exploratorio de Datos (EDA), entrenamiento comparativo de modelos de clasificación y evaluación sobre el dataset **WineQT**.

## Estructura del Proyecto

```text
KUGM_EDA_Vinos/
├── data/
│   ├── WineQT.csv          # Dataset original
│   ├── wine_clean.csv      # Dataset limpio generado por eda.py
│   ├── X_test.csv          # Características de prueba
│   └── y_test.csv          # Etiquetas de prueba
├── models/
│   └── mejor_modelo.joblib # Pipeline ganador serializado
├── outputs/
│   ├── 1_histogramas.png
│   ├── 2_boxplots.png
│   ├── 3_matriz_correlacion.png
│   ├── 4_comparacion_modelos.png
│   └── 5_matriz_confusion.png
├── src/
│   ├── eda.py              # Carga, limpieza, estadísticas y visualización
│   ├── entrenamiento.py    # Validación cruzada (4 modelos) y guardado del ganador
│   └── pruebas.py          # Evaluación final y matriz de confusión en test
├── .gitignore
├── requirements.txt
└── README.md
```

## Requisitos e Instalación

* **Python 3.10** o superior.
* Dependencias incluidas en `requirements.txt` (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`).

1. **Clonar el repositorio y entrar al directorio del proyecto:**
   ```powershell
   git clone [https://github.com/](https://github.com/)<TU_USUARIO>/KUGM_EDA_Vinos.git
   cd KUGM_EDA_Vinos
   ```

2. **Crear y activar el entorno virtual en Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instalar las dependencias:**
   ```powershell
   pip install -r requirements.txt
   ```

## Orden de Ejecución

Ejecuta los módulos de forma secuencial desde la carpeta raíz (`KUGM_EDA_Vinos/`):

1. **Análisis Exploratorio de Datos y Limpieza:**
   Carga `data/WineQT.csv`, muestra estadísticas descriptivas, elimina la columna `Id` y registros duplicados, guarda las gráficas exploratorias en `outputs/` y genera `data/wine_clean.csv`.
   ```powershell
   python src/eda.py
   ```

2. **Entrenamiento y Selección del Mejor Modelo:**
   Divide el dataset (80% entrenamiento / 20% prueba), compara 4 modelos mediante validación cruzada estratificada, exporta la gráfica comparativa en `outputs/` y guarda el pipeline ganador en `models/mejor_modelo.joblib`.
   ```powershell
   python src/entrenamiento.py
   ```

3. **Evaluación Final en Datos de Prueba:**
   Carga `models/mejor_modelo.joblib`, evalúa las predicciones sobre el conjunto inédito (`data/X_test.csv` y `data/y_test.csv`), muestra las métricas en consola y guarda la matriz de confusión en `outputs/5_matriz_confusion.png`.
   ```powershell
   python src/pruebas.py
   ```