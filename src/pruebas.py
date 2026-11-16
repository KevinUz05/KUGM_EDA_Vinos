from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"
MODEL_PATH = BASE_DIR / "models" / "mejor_modelo.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"


def cargar_modelo(ruta: Path):
    """Carga el pipeline entrenado desde disco."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta primero 'python src/entrenamiento.py'."
        )
    modelo = joblib.load(ruta)
    print(f"[OK] Modelo cargado correctamente desde: {ruta}")
    return modelo


def cargar_datos_prueba(x_path: Path, y_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga las características y etiquetas del conjunto de prueba."""
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "No se encontraron X_test.csv o y_test.csv. Ejecuta primero 'python src/entrenamiento.py'."
        )
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path)["quality"]
    print(f"[OK] Datos de prueba cargados: {len(X_test)} registros.")
    return X_test, y_test


def evaluar_modelo(
    modelo, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path
) -> None:
    """Genera predicciones, imprime métricas finales y guarda la matriz de confusión."""
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n=== MÉTRICAS FINALES EN CONJUNTO DE PRUEBA (TEST) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\n=== REPORTE DE CLASIFICACIÓN POR CLASE (QUALITY) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Muestra de predicciones individuales
    comparacion = pd.DataFrame(
        {
            "Real": y_test.values[:10],
            "Predicho": y_pred[:10],
            "Acierto": ["Sí" if r == p else "No" for r, p in zip(y_test.values[:10], y_pred[:10])],
        }
    )
    print("=== MUESTRA DE PREDICCIONES (PRIMEROS 10 REGISTROS) ===")
    print(comparacion.to_string(index=False))

    # Matriz de confusión
    output_dir.mkdir(parents=True, exist_ok=True)
    etiquetas = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=etiquetas)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=etiquetas,
        yticklabels=etiquetas,
    )
    plt.title("Matriz de Confusión - Evaluación en Test", fontsize=13)
    plt.xlabel("Calidad Predicha")
    plt.ylabel("Calidad Real")
    plt.tight_layout()
    ruta_cm = output_dir / "5_matriz_confusion.png"
    plt.savefig(ruta_cm, dpi=150)
    plt.close()

    print(f"\n[OK] Matriz de confusión guardada en: {ruta_cm}")


def main() -> None:
    modelo = cargar_modelo(MODEL_PATH)
    X_test, y_test = cargar_datos_prueba(X_TEST_PATH, Y_TEST_PATH)
    evaluar_modelo(modelo, X_test, y_test, OUTPUT_DIR)


if __name__ == "__main__":
    main()