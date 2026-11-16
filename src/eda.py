from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WineQT.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "wine_clean.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta}")
    df = pd.read_csv(ruta)
    print(f"[OK] Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas.\n")
    return df


def mostrar_estadisticas(df: pd.DataFrame) -> None:
    """Imprime información general, nulos, duplicados y estadísticas descriptivas."""
    print("=== INFORMACIÓN GENERAL ===")
    print(df.info())

    print("\n=== VALORES NULOS POR COLUMNA ===")
    print(df.isnull().sum())

    print(f"\n=== FILAS DUPLICADAS: {df.duplicated().sum()} ===")

    print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
    print(df.describe().T.round(3))

    print("\n=== DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (quality) ===")
    print(df["quality"].value_counts().sort_index())


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset eliminando duplicados, nulos y la columna Id."""
    df_limpio = df.copy()

    if "Id" in df_limpio.columns:
        df_limpio = df_limpio.drop(columns=["Id"])
        print("\n[LIMPIEZA] Columna 'Id' eliminada.")

    antes = len(df_limpio)
    df_limpio = df_limpio.dropna().drop_duplicates()
    despues = len(df_limpio)
    print(f"[LIMPIEZA] Registros eliminados (nulos/duplicados): {antes - despues}")
    print(f"[LIMPIEZA] Dataset final listo: {df_limpio.shape[0]} filas y {df_limpio.shape[1]} columnas.")

    return df_limpio


def visualizar_distribuciones(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera y guarda histogramas, boxplots y matriz de correlación."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Histogramas de todas las variables
    axes = df.hist(bins=20, figsize=(15, 11), color="#722F37", edgecolor="white")
    plt.suptitle("Distribución de Variables Fisicoquímicas y Calidad del Vino", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "1_histogramas.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 2. Boxplots de variables predictoras (estandarizadas para comparar escalas)
    features = df.drop(columns=["quality"])
    features_norm = (features - features.mean()) / features.std()
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=features_norm, orient="h", palette="flare")
    plt.title("Boxplots de Variables Predictoras (Valores Estandarizados)", fontsize=13)
    plt.xlabel("Desviaciones estándar respecto a la media")
    plt.tight_layout()
    plt.savefig(output_dir / "2_boxplots.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 3. Matriz de correlación
    plt.figure(figsize=(12, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Matriz de Correlación - WineQT", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "3_matriz_correlacion.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n[OK] Gráficas guardadas en: {output_dir}")


def main() -> None:
    df = cargar_datos(DATA_PATH)
    mostrar_estadisticas(df)
    df_limpio = limpiar_datos(df)
    visualizar_distribuciones(df_limpio, OUTPUT_DIR)

    df_limpio.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"[OK] Dataset limpio guardado en: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()