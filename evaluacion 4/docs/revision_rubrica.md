# Revisión frente a la rúbrica

Esta revisión identifica evidencia y pendientes; no asigna una nota ni garantiza la calificación del docente.

| Criterio de la rúbrica | Evidencia | Verificación y límites |
|---|---|---|
| Comprensión del problema y dataset | Sección 1 y diccionario en 2 | Contexto, objetivo, preguntas y tipos de variables |
| EDA y visualizaciones exploratorias | Sección 2 y figuras 01–08 | Distribuciones, faltantes, duplicados, texto, categorías y correlaciones |
| Preprocesamiento de datos | Sección 3 | Limpieza, TF-IDF, indicadores, One-Hot y escalado dentro de pipelines |
| Variables y reducción de dimensionalidad | Secciones 3, 5 y 6 | Descarte justificado de ID, alta cardinalidad, SVD y proyecciones; varianza descartada no se supone inútil |
| Algoritmos supervisados | Sección 4 | Regresión logística y SVM, además de referencias tabulares; entrenamiento y optimización ejecutados |
| Algoritmos no supervisados | Sección 5 | K-Means y Ward entrenados y evaluados en una misma muestra sin etiquetas; NMF e Isolation Forest complementarios |
| Comparación y selección de modelos | Secciones 4 y 7 | Selección por AP de validación, test al final; umbral F1 OOF |
| Uso e interpretación de métricas | Secciones 4–7 y results/ | Precision, Recall, F1, matriz, AP/ROC-AUC, silhouette, DB y CH; contraste con etiquetas después del agrupamiento |
| Interpretación y conclusiones | Secciones 7–9 | Respuestas a las ocho preguntas, límites y recomendaciones; cifras actualizadas desde la ejecución |
| Trabajo en equipo y GitHub | docs/colaboracion.md e historial Git | Los commits técnicos usan la identidad configurada; falta evidencia de contribuciones individuales reales de los compañeros |
| Presentación o video | video/guion.md y video/notas_presentacion.md | Material preparado; grabación, participación de todos y enlace pendientes |

## Verificación técnica realizada

Se descargó el CSV indicado en la guía y se ejecutó el notebook completo desde cero. Las métricas y la huella del dataset se guardan en `results/resumen.json`; las tablas se exportan a CSV. Las versiones exactas se guardan en `requirements-lock.txt`. Se comprueba coherencia de los conteos de la matriz y sus métricas, ejecución sin errores y sincronía del notebook con el script.

## Límites metodológicos que deben explicarse

- Esta versión reserva prueba antes del EDA y calcula la exploración en entrenamiento. El corpus ya había sido examinado en una versión anterior: para una evaluación externa rigurosa conviene obtener otro conjunto independiente.
- El umbral se obtiene después de seleccionar hiperparámetros sobre los mismos datos de entrenamiento. La validación anidada reduce el optimismo de la estimación OOF.
- Agrupar por texto exacto no separa necesariamente empresas o plantillas similares. Las métricas son válidas para esta partición y no garantizan generalización a otro idioma o periodo.
- La muestra de agrupamiento y la selección de dimensiones influyen en los resultados. Las etiquetas se consultan únicamente después de ajustar la representación y los grupos.

## Pendientes humanos antes de entregar

1. Confirmar nombres completos y aportes de cada integrante; todos deben revisar y sustentar el trabajo.
2. Registrar aportes reales de cada compañero en GitHub. No simular autorías ni participación.
3. Grabar el video de máximo 12 minutos, incluir a los cuatro integrantes y completar `video/enlace_video.md`.
4. Verificar que el docente tiene acceso al repositorio y al video. La guía solicita usar el repositorio del curso; el usuario proporcionó el repositorio de esta entrega.
