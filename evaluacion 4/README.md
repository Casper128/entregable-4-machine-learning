# Evento Evaluativo 4 — Equipo 10

**Estado:** revisión técnica realizada; falta ejecutar de nuevo con el CSV. Las cifras siguientes pertenecen a la ejecución original y deben contrastarse con la versión corregida. Ver [revisión de rúbrica](docs/revision_rubrica.md) y [colaboración](docs/colaboracion.md).
## Detección de ofertas laborales fraudulentas y análisis de patrones textuales

**Asignatura:** Análisis de Datos · **Docente:** Daniel Alexis Nieto Mora · **Programa:** Ingeniería en Ciencia de Datos · **Grupo:** 190304018-1 · **Periodo:** 2026-2

### Integrantes
| Nombre | Usuario GitHub | Responsabilidad principal |
|---|---|---|
| Por confirmar | @dariomontoya0505 | Aporte por confirmar |
| Por confirmar | @Franyelica | Aporte por confirmar |
| Por confirmar | @santiagopabon15 | Aporte por confirmar |
| Darwin | @Casper128 | Aporte por confirmar |

---

### 1. Descripción del problema
Las plataformas de empleo se usan para publicar ofertas falsas que buscan robar datos personales o estafar a los
solicitantes. Con el dataset *Real or Fake Job Posting Prediction* (≈17.9 mil publicaciones, ≈4.8 % fraudulentas)
desarrollamos dos componentes:

* **Supervisado:** clasificar ofertas como legítimas o fraudulentas comparando tres enfoques: solo variables
  estructuradas, solo texto (TF-IDF) y texto + estructuradas.
* **No supervisado:** descubrir temáticas en las publicaciones con TF-IDF + LSA + agrupamiento (K-Means y
  aglomerativo Ward, más NMF e Isolation Forest como complemento) **sin usar la etiqueta**, y analizar después si
  algún grupo concentra fraude.

### 2. Fuente de datos
* Kaggle — *Real or Fake: Fake JobPosting Prediction*: <https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction>
* Origen: EMSCAD (Employment Scam Aegean Dataset), Universidad del Egeo.
* Archivo usado: `fake_job_postings.csv` (no se versiona en el repositorio por su tamaño; ver instrucciones).

### 3. Estructura del proyecto
```
evaluacion 4/
├── README.md
├── requirements.txt
├── evento4_fake_jobs.ipynb     # notebook principal (resultados originales; requiere reejecución)
├── evento4_fake_jobs.py        # mismo notebook en formato script (jupytext, cómodo para revisar diffs)
├── data/
│   └── fake_job_postings.csv   # descargar de Kaggle (no incluido)
├── figures/                    # figuras generadas automáticamente por el notebook
└── video/
    └── enlace_video.md         # enlace al video de presentación (máx. 12 min)
```

### 4. Instrucciones de ejecución
```bash
pip install -r requirements.txt
# descargar fake_job_postings.csv de Kaggle y copiarlo en data/
jupyter notebook evento4_fake_jobs.ipynb   # Kernel → Restart & Run All
```
Tiempo aproximado de ejecución completa: 12–25 min en un portátil (la búsqueda de hiperparámetros es la parte más
costosa). Todas las semillas están fijadas (`SEED = 42`) para que los resultados sean reproducibles.

### 5. Metodología (resumen)
1. **EDA:** desbalance, faltantes por clase (son informativos), cardinalidad, tasas de fraude por categoría,
   longitud del texto, términos distintivos y correlaciones.
2. **Preprocesamiento:** limpieza de texto, TF-IDF de 1-2 gramas, indicadores de faltantes, país extraído de
   `location`, One-Hot con agrupación de categorías raras, `log1p` + escalado. **Partición 80/20 estratificada
   antes de ajustar cualquier transformación; todo dentro de `Pipeline`** para evitar fuga de información.
3. **Supervisado:** Regresión Logística, SVM lineal, Random Forest y HistGradientBoosting; comparación de
   estrategias de desbalance (sin tratamiento, ponderación de clases, submuestreo y ajuste de umbral);
   `GridSearchCV` optimizando PR-AUC; evaluación final única en test con Precision, Recall, F1, PR-AUC, ROC-AUC y
   matriz de confusión; interpretación de coeficientes e importancias; análisis de errores.
4. **No supervisado:** TF-IDF → LSA (100 componentes) → K-Means (k elegido con Silhouette, Davies-Bouldin,
   Calinski-Harabasz y codo) y aglomerativo Ward; estabilidad (ARI entre semillas); perfiles de grupos con
   términos representativos y ejemplos; tópicos NMF; Isolation Forest.
5. **Visualización:** proyecciones SVD, t-SNE y UMAP coloreadas por grupo y por etiqueta.
6. **Análisis comparativo** supervisado vs. no supervisado (8 preguntas de la guía) y conclusiones.

### 6. Principales resultados
**Datos:** 17 880 ofertas → 17 599 tras eliminar 281 duplicados exactos; 4,86 % fraudulentas. Partición 80/20
estratificada y **agrupada por texto** (las ~1 800 ofertas con texto repetido nunca quedan repartidas entre
entrenamiento y prueba).

**Componente supervisado — conjunto de prueba (3 520 ofertas, 171 fraudes)**

| Modelo | Enfoque | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Random Forest | solo estructuradas | 0,825 | 0,982 | 0,663 | 0,749 | 0,703 |
| Regresión Logística | texto + estructuradas | 0,924 | 0,992 | 0,838 | 0,877 | 0,857 |
| **SVM lineal (seleccionado)** | texto + estructuradas | **0,931** | 0,991 | 0,815 | 0,877 | 0,845 |

* El modelo seleccionado detecta **150 de 171 fraudes** con solo 34 falsas alarmas entre 3 349 ofertas legítimas (1 %).
* El **texto** es la fuente principal de señal (PR-AUC en validación cruzada: 0,91 con texto vs 0,80 con
  estructuradas).
* La mejor estrategia de desbalance fue **ponderar las clases y ajustar el umbral**; el submuestreo empeoró el
  modelo (PR-AUC 0,64).
* Señales de fraude: perfil de empresa vacío, sin logo, promesas de dinero (*earn, money, income*), "apply using
  link", trabajos de entrada / data entry / desde casa, industria de petróleo y energía.

**Componente no supervisado**
* TF-IDF → LSA (100 componentes) → K-Means con k = 11 (silhouette 0,105) y aglomerativo Ward (silhouette 0,081).
* Los grupos describen **temas** y no la etiqueta (ARI ≈ 0), pero el fraude se concentra: los temas
  *administración / oficina* (16,8 % de fraude, 3,5× la tasa global) y *atención al cliente / trabajo desde casa*
  (10,1 %) reúnen ≈ 30 % de las ofertas y ≈ 78 % de los fraudes. Plantillas de agencias (TEFL, aprendices) forman
  grupos con 0 % de fraude.
* Isolation Forest muestra que el fraude **no es una anomalía textual** (PR-AUC 0,10): las ofertas falsas imitan a
  las reales.

**Recomendación:** usar la SVM lineal / Regresión Logística como filtro de priorización con revisión humana y
revisión reforzada en los temas de alto riesgo. Limitaciones y detalle en las secciones 7 y 8 del notebook.

### 7. Video
Enlace: _(completar)_ — ver `video/enlace_video.md`.
