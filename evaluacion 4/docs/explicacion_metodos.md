# Apoyo para comprender y sustentar el trabajo

## Del texto a los números

Cada publicación reúne título, perfil, descripción, requisitos y beneficios. La limpieza uniforma mayúsculas, elimina HTML y conserva indicadores de enlaces, correos y teléfonos. No se traduce: las palabras vacías usadas son del inglés y se reconoce la limitación de idioma.

TF-IDF asigna peso a las palabras: un término frecuente en una oferta pero poco común en el corpus recibe mayor importancia. Los bigramas representan pares como “data entry”. El vocabulario y los pesos IDF se aprenden únicamente con entrenamiento dentro de cada pliegue supervisado. No se usa la etiqueta como característica.

## Clasificación: métodos vistos en clase

La regresión logística combina las variables para estimar la probabilidad de fraude. Su coeficiente positivo indica asociación con mayor puntaje, condicionado por el resto de características; no demuestra causalidad ni es evidencia suficiente para acusar una empresa. La ponderación de clases altera el entrenamiento y sus probabilidades no deben suponerse calibradas sin evaluación adicional.

La SVM lineal busca una frontera con margen amplio. Se permite cometer errores mediante regularización: no exige separabilidad perfecta. `C` regula el costo de esos errores. Su puntaje de decisión no es una probabilidad.

Random Forest e HistGradientBoosting son referencias complementarias para datos estructurados. La selección principal compara regresión logística y SVM, conectando el desarrollo con las semanas 12–13.

## Evaluación y desbalance

Precision = TP/(TP+FP): de las alertas emitidas, cuántas corresponden a fraude. Recall = TP/(TP+FN): de todos los fraudes, cuántos se detectaron. F1 combina ambas mediante media armónica. La matriz de confusión hace visibles los costos prácticos de los errores.

AP resume el ordenamiento de los casos al recorrer los umbrales de precisión-recall. No es un porcentaje de ofertas clasificadas correctamente. Se conserva el término PR-AUC usado en tablas, precisando que la implementación calcula Average Precision.

Se elige modelo e hiperparámetros con validación de entrenamiento. El umbral se obtiene con predicciones OOF, es decir, predicciones realizadas por modelos que no entrenaron con ese pliegue. La prueba se consulta al final. No deben cambiarse parámetros para mejorar sus resultados después de verla.

## Agrupamiento y reducción de dimensiones

SVD comprime TF-IDF sin convertir toda la matriz dispersa en una matriz densa. LSA es el uso de esta descomposición para representar temas textuales. Se relaciona con PCA por su propósito de reducir dimensiones, pero TruncatedSVD no centra las variables como PCA.

K-Means resume documentos en centroides y requiere elegir k. Ward une grupos minimizando el incremento de variación interna y permite contrastar una estructura jerárquica. Ambos reciben exclusivamente representaciones textuales. La etiqueta se agrega después para describir las tasas de fraude de los grupos.

Silhouette mayor, Davies-Bouldin menor y Calinski-Harabasz mayor sugieren mejores particiones en un espacio dado. No garantizan utilidad semántica. Una tasa de fraude alta dentro de un tema no significa que todas sus ofertas sean fraudulentas.

t-SNE permite observar vecindarios locales. Las distancias entre islas y sus tamaños no deben interpretarse como distancias reales entre temas. NMF e Isolation Forest son complementos exploratorios; no sustituyen la comparación principal K-Means/Ward.

## Preguntas que todos deben poder responder

- ¿Por qué una accuracy cercana al 95 % podría ser inútil aquí?
- ¿Qué diferencia hay entre elegir el modelo con validación y evaluarlo con prueba?
- ¿Por qué los textos duplicados deben permanecer en el mismo grupo al dividir datos?
- ¿Por qué un cluster puede mezclar ofertas legítimas y fraudulentas?
- ¿Qué consecuencias tienen un falso positivo y un falso negativo?
- ¿Qué limitaciones impiden usar estas predicciones como prueba definitiva de fraude?
