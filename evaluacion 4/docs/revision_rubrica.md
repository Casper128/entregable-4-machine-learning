# Revisión frente a la rúbrica

No se asigna una nota: la calificación depende del docente y de la sustentación.

| Criterio | Evidencia en notebook | Estado y acción |
|---|---|---|
| Comprensión del problema y dataset | 1 y diccionario de variables | Cubierto; explicar moderación de ofertas |
| EDA y visualizaciones | 2, figuras 01–08 | Cubierto en ejecución original; revisar cifras al ejecutar |
| Preprocesamiento | 3, pipelines, grupos por texto | Implementado; reconocer EDA previo al test como limitación |
| Variables y dimensionalidad | 3, 5 y 6 | Justificado; SVD adaptado a matrices dispersas |
| Modelos supervisados | 4: LR, SVM y referencias tabulares | Implementado; selección corregida para usar CV |
| Modelos no supervisados | 5: K-Means y Ward | Implementado; explicitar diferencias de muestras de entrenamiento |
| Comparación y selección | 4.5–4.7, 7 | Reejecutar y actualizar recomendación según CV |
| Métricas | Precision, Recall, F1, matriz, AP y métricas de grupos | Interpretadas; precisiones añadidas en 9 |
| Interpretación y conclusiones | 7–9 | Contrastar afirmaciones y valores con nueva ejecución |
| Equipo, GitHub y commits | docs/colaboracion.md | Pendiente de aportes verificables y repositorio del curso |
| Presentación/video | video/guion.md | Guion preparado; grabación y enlace pendientes |

## Antes de entregar

1. Descargar el CSV original y colocarlo en `data/fake_job_postings.csv`.
2. Instalar requirements y reiniciar/ejecutar todo el notebook desde cero.
3. Revisar advertencias, tablas, matriz de confusión y coherencia de cifras del README y conclusiones. La selección por CV puede cambiar el modelo recomendado.
4. Confirmar nombres, usuario de Darwin y aportes de cada integrante.
5. Incorporar esta carpeta al mismo repositorio del curso y registrar contribuciones reales.
6. Grabar el video con todos los integrantes y completar el enlace.

No se pudo repetir el entrenamiento durante esta revisión porque el CSV no está en la carpeta proporcionada. Los resultados guardados no sustituyen esa verificación.
