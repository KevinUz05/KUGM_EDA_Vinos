from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DATA_PATH = DATA_DIR / "wine_clean.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


def cargar_datos_limpios(ruta: Path) -> pd.DataFrame:
    """Carga el dataset limpio generado en la etapa de EDA."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Ejecuta primero 'python src/eda.py'."
        )
    df = pd.read_csv(ruta)
    print(f"[OK] Dataset limpio cargado: {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df


def preparar_y_dividir_datos(
    df: pd.DataFrame, data_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa características y objetivo, divide en train/test y guarda el set de prueba."""
    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_test.to_csv(data_dir / "y_test.csv", index=False)

    print(f"[OK] División 80/20 completada -> Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[OK] Conjunto de prueba guardado en: {data_dir}")
    return X_train, X_test, y_train, y_test


def entrenar_y_comparar(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path
) -> tuple[str, Pipeline, pd.DataFrame]:
    """Evalúa 5 modelos distintos mediante validación cruzada y entrena el mejor."""
    modelos = {
        "Regresión Logística": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.5, probability=True, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(n_estimators=150, random_state=42),
                ),
            ]
        ),
    }

    n_splits = min(5, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall": make_scorer(recall_score, average="weighted", zero_division=0),
        "f1": make_scorer(f1_score, average="weighted", zero_division=0),
    }

    resultados = []
    print(f"\n=== EVALUACIÓN CON VALIDACIÓN CRUZADA ({n_splits} FOLDS) ===")

    for nombre, pipeline in modelos.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        resultados.append(
            {
                "Modelo": nombre,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1-Score": scores["test_f1"].mean(),
            }
        )

    df_resultados = (
        pd.DataFrame(resultados)
        .set_index("Modelo")
        .sort_values(by="F1-Score", ascending=False)
    )
    print("\n=== COMPARACIÓN DE MODELOS (ORDENADOS POR F1-SCORE) ===")
    print(df_resultados.round(4))

    # Generar gráfica comparativa
    output_dir.mkdir(parents=True, exist_ok=True)
    df_resultados.plot(kind="bar", figsize=(11, 5.5), rot=0, colormap="viridis")
    plt.title("Comparación de 5 Modelos de Clasificación (Validación Cruzada)", fontsize=13)
    plt.ylabel("Puntaje promedio")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "4_comparacion_modelos.png", dpi=150)
    plt.close()

    # Seleccionar y entrenar el ganador con todo X_train
    mejor_nombre = df_resultados["F1-Score"].idxmax()
    mejor_pipeline = modelos[mejor_nombre]
    mejor_pipeline.fit(X_train, y_train)

    print(
        f"\n[GANADOR] '{mejor_nombre}' seleccionado (F1-Score CV: "
        f"{df_resultados.loc[mejor_nombre, 'F1-Score']:.4f}) y entrenado con X_train completo."
    )
    return mejor_nombre, mejor_pipeline, df_resultados


def guardar_modelo(modelo: Pipeline, models_dir: Path) -> Path:
    """Serializa y guarda el pipeline ganador en disco."""
    models_dir.mkdir(parents=True, exist_ok=True)
    ruta_modelo = models_dir / "mejor_modelo.joblib"
    joblib.dump(modelo, ruta_modelo)
    print(f"[OK] Modelo guardado exitosamente en: {ruta_modelo}")
    return ruta_modelo


def main() -> None:
    df = cargar_datos_limpios(CLEAN_DATA_PATH)
    X_train, _, y_train, _ = preparar_y_dividir_datos(df, DATA_DIR)
    _, mejor_modelo, _ = entrenar_y_comparar(X_train, y_train, OUTPUT_DIR)
    guardar_modelo(mejor_modelo, MODELS_DIR)


if __name__ == "__main__":
    main()