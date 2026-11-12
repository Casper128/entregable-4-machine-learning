# Evento Evaluativo 4 — Equipo 10

Detección de ofertas laborales fraudulentas y análisis de patrones textuales.

Asignatura: Análisis de Datos. Docente: Daniel Alexis Nieto Mora. Ingeniería en Ciencia de Datos, periodo 2026-2.

## Integrantes

- @Casper128 (Darwin).
- @dariomontoya0505.
- @Franyelica.
- @santiagopabon15.

## Problema y datos

Buscamos clasificar ofertas legítimas y fraudulentas, y descubrir temas mediante agrupamiento sin etiquetas. Fuente: [Real or Fake Job Posting Prediction, Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). El CSV original contiene 17 880 ofertas y 866 fraudes; tras eliminar duplicados exactos quedan 17599 ofertas (4.86% fraudulentas).

## Ejecución

Desde esta carpeta, usando Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python descargar_datos.py
jupyter notebook evento4_fake_jobs.ipynb
```

En Windows activar el entorno con `.venv\Scripts\activate`. En Jupyter elegir el kernel del entorno y ejecutar “Restart Kernel and Run All Cells”. La búsqueda de parámetros y las proyecciones pueden tardar varios minutos; el tiempo depende del equipo. El script `.py` es una representación con celdas `# %%`, útil para revisar cambios; el notebook es la entrega principal. Las advertencias de convergencia deben revisarse.

## Metodología

1. Reserva de prueba por grupos antes del EDA; distribuciones, faltantes, texto, duplicados, correlaciones y categorías en entrenamiento.
2. Limpieza, TF-IDF, codificación y escalado en pipelines; separación de entrenamiento/prueba agrupada por textos repetidos.
3. Comparación de regresión logística, SVM y referencias tabulares; ponderación, submuestreo y umbral OOF. Selección por AP de validación.
4. TF-IDF y SVD; K-Means y Ward comparados en la misma muestra aleatoria sin etiquetas. Perfiles en el corpus completo y complementos NMF/Isolation Forest.
5. Visualizaciones SVD, t-SNE y UMAP; comparación de enfoques y limitaciones.

## Resultados verificados

Modelo seleccionado por validación: **LinearSVC** (AP CV 0.875). En prueba: 3520 ofertas, 171 fraudes; TP=151, FN=20, FP=33, TN=3316.

| modelo | umbral | precision | recall | f1 | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest (estructurado) | umbral por defecto | 0.660 | 0.737 | 0.696 | 0.824 | 0.981 |
| LogReg (texto+estructurado) | umbral por defecto | 0.860 | 0.860 | 0.860 | 0.922 | 0.991 |
| LogReg (texto+estructurado) | umbral ajustado (OOF) | 0.898 | 0.825 | 0.860 | 0.922 | 0.991 |
| LinearSVC (texto+estructurado) | umbral por defecto | 0.907 | 0.860 | 0.883 | 0.930 | 0.992 |
| LinearSVC (texto+estructurado) | umbral ajustado (OOF) | 0.821 | 0.883 | 0.851 | 0.930 | 0.992 |

Se llama PR-AUC a AP en las tablas; se calcula con `average_precision_score`, no integración trapezoidal. El umbral se elige para maximizar F1 OOF, no recall explícitamente.

Agrupamiento: k=11; silhouette en muestra: K-Means 0.121, Ward 0.107. SVD conserva 27.9% de la varianza con 100 componentes. Estos resultados muestran separación limitada; los grupos se interpretan con términos y ejemplos, no como detectores definitivos de fraude.

**Recomendación:** priorizar revisión humana con el clasificador; usar los temas como contexto. Límites: antigüedad e idioma del corpus, posibles etiquetas imperfectas, memorización de empresas, exploración histórica del mismo corpus en una versión previa y selección de parámetros previa a OOF. Consultar secciones 7–9 del notebook.

## Archivos

- `evento4_fake_jobs.ipynb`: desarrollo ejecutado y resultados.
- `evento4_fake_jobs.py`: celdas equivalentes para revisión.
- `descargar_datos.py` y `data/README.md`: obtención y huella del CSV; datos fuera de Git.
- `results/`: métricas, perfiles, selección k y resumen de ejecución.
- `figures/`: figuras regeneradas.

## Verificación

Notebook ejecutado desde cero con el CSV cuya huella SHA-256 es `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`. Versiones y métricas están en `results/resumen.json`; dependencias exactas en `requirements-lock.txt`. Las cifras corresponden a la ejecución completa del notebook.
