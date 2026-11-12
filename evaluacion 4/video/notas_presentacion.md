# Notas para la presentación del equipo

Guía para ensayar con palabras propias. Repartir los cuatro bloques según aportes reales; cada integrante debe entender el proceso completo. Seguir los tiempos de `guion.md`, sin leer todas las tablas ni el código.

## Bloque 1: problema y exploración (aproximadamente 3 minutos)

Nuestro objetivo es apoyar la revisión de ofertas laborales. Una oferta fraudulenta puede buscar datos personales o dinero de quien está buscando trabajo. El proyecto tiene dos preguntas: si podemos detectar una oferta sospechosa a partir de ejemplos conocidos y qué temas o patrones aparecen entre las publicaciones sin consultar sus etiquetas.

Usamos Real or Fake Job Posting Prediction de Kaggle. El archivo contiene 17 880 registros y 18 columnas. Hay texto libre, categorías y variables binarias, además de la etiqueta de fraude. Retiramos duplicados exactos y quedaron 17599 ofertas, con 4.86% de fraude.

Mostrar la distribución de clases. Explicar que responder siempre “legítima” tendría una accuracy elevada sin detectar ningún fraude. Mostrar faltantes y comentar que una ausencia puede aportar información predictiva, pero una asociación no demuestra causalidad. Los textos repetidos se mantienen juntos al separar entrenamiento y prueba para reducir memorización.

## Bloque 2: preparación y clasificación (aproximadamente 3 minutos)

Unimos título, perfil de empresa, descripción, requisitos y beneficios. Limpiamos HTML y normalizamos el texto; representamos palabras y pares de palabras con TF-IDF. Esta técnica destaca términos característicos de cada documento en relación con el corpus. Codificamos categorías y escalamos variables numéricas. Vocabulario, pesos y transformaciones se ajustan dentro del pipeline usando solo entrenamiento en cada pliegue.

Los modelos principales son regresión logística y SVM lineal, vistos en el curso. También usamos árboles con variables estructuradas como referencia. Comparamos texto, estructura y combinación, además de ponderación y submuestreo. Elegimos mediante validación de entrenamiento y ajustamos el umbral para maximizar F1 OOF, sin elegirlo a partir del test.

El modelo seleccionado fue LinearSVC. En prueba se analizaron 3520 ofertas, de las cuales 171 eran fraudulentas. Detectó 151 fraudes, omitió 20 y generó 33 falsas alarmas. Precision fue 0.821, Recall 0.883, F1 0.851 y AP 0.930. Explicar esos números usando la matriz. El puntaje SVM no es una probabilidad; AP tampoco es un porcentaje de aciertos.

## Bloque 3: agrupamiento y visualización (aproximadamente 3 minutos)

En el componente no supervisado excluimos la etiqueta. TF-IDF produce muchas dimensiones; usamos SVD para comprimir la representación a 100 componentes, que conservan 27.9% de la varianza. Esto facilita el cálculo, pero también descarta información potencialmente útil.

Comparamos K-Means y agrupamiento aglomerativo Ward en la misma muestra aleatoria. El primero busca centroides y el segundo construye agrupaciones jerárquicas minimizando variación interna. Elegimos 11 grupos para el K-Means temático mediante criterios internos y el propósito de interpretación. No seleccionamos grupos por su tasa de fraude.

Mostrar los términos y ejemplos de dos grupos contrastantes de `results/perfiles_kmeans.csv`, sin asumir que todos los avisos de un tema son fraudulentos. Después del agrupamiento calculamos tasas por etiqueta. Las métricas muestran separación limitada: los temas se solapan. NMF e Isolation Forest sirven como exploraciones adicionales.

Mostrar la proyección t-SNE. Explicar que ayuda a explorar vecinos locales; la distancia entre islas no permite medir directamente la separación real entre temas. Una figura atractiva no reemplaza la evaluación del agrupamiento.

## Bloque 4: comparación y conclusiones (aproximadamente 2 minutos y 45 segundos)

Los enfoques responden preguntas diferentes. La clasificación aprende la relación entre características y fraude; el agrupamiento describe temas. Un grupo puede mezclar ofertas legítimas y falsas porque trata sobre un mismo tipo de empleo. Por eso los grupos no reemplazan al detector supervisado.

Recomendamos un sistema de priorización con revisión humana. Un falso negativo puede exponer a una persona a una estafa; un falso positivo retrasa una oferta legítima. La política del umbral debe relacionarse con esos costos y con la capacidad de revisión. En este experimento se optimizó F1, no se estimaron costos económicos.

Reconocer las limitaciones: corpus histórico y principalmente en inglés, etiquetas imperfectas y posible memorización de empresas. La versión actual reserva prueba antes del EDA y explora solo entrenamiento; como hubo una exploración previa del corpus, conviene validar además con datos externos. También se puede mejorar la evaluación del umbral con validación anidada. No presentamos los resultados como garantía para plataformas actuales o publicaciones en español.

Cerrar explicando lo aprendido: preparar datos y justificar la evaluación es tan importante como entrenar algoritmos. Las conclusiones se apoyan en métricas y ejemplos, y reconocen dónde todavía hace falta validación.
