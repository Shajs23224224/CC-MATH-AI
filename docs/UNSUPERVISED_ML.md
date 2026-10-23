# F30 — Unsupervised Machine Learning

## Estado y límites

Esta fase añade una API allow-listed para clustering, reducción de dimensionalidad y detección
de anomalías. La implementación propuesta se valida mediante el workflow F30 y el CI general;
no se considera integrada en `main` hasta que la PR se revise y se fusione. Los tests usan datos
sintéticos para comprobar contratos y límites de información; no son benchmarks de mercado.

F30 no reemplaza F26 (detección de regímenes), no asigna significado económico automático a
los grupos ni convierte anomalías en señales de inversión.

## Algoritmos

| Operación | Identificadores | Salida |
| --- | --- | --- |
| Clustering | `kmeans`, `agglomerative_clustering`, `dbscan`, `gaussian_mixture` | Etiquetas del prefijo de entrenamiento y, si el estimador admite inferencia inductiva, etiquetas del sufijo reservado |
| Reducción | `pca`, `kernel_pca`, `fast_ica` | Embeddings de entrenamiento y del sufijo reservado |
| Anomalías | `isolation_forest`, `local_outlier_factor`, `elliptic_envelope` | Scores y banderas `1` (inlier) / `-1` (outlier) para entrenamiento y holdout |

Se reutilizan NumPy, pandas indirectamente cuando lo exige el proyecto, scikit-learn y los contratos
Pydantic existentes. No se añade una dependencia externa para esta fase.

## Separación de datos y escalado

La API recibe una matriz numérica ordenada por el llamador y nombres de variables únicos. La
secuencia se divide de forma cronológica: el prefijo inicial se usa para ajustar `StandardScaler`
y el estimador; el sufijo final queda reservado para transformaciones o inferencia que el
algoritmo soporte. El scaler jamás se ajusta sobre el sufijo. La huella SHA-256 se calcula sobre
las observaciones y nombres del prefijo; por tanto, cambiar sólo las filas reservadas no cambia
la huella ni los parámetros del scaler.

Todos los valores de entrada deben ser finitos. La API exige un mínimo de ocho observaciones de
entrenamiento, dos de holdout y doce en total; el máximo predeterminado es 5.000 filas y puede
ajustarse dentro del límite explícito del contrato. Los parámetros de estimadores se validan
contra su lista real de scikit-learn; semilla y paralelismo permanecen bajo control del servicio
(`random_state` no se acepta como parámetro libre y `n_jobs` queda fijado en uno).

DBSCAN y clustering jerárquico no ofrecen una predicción inductiva estándar para observaciones
nuevas en esta API. Su salida `holdout_labels` queda en `null`, en vez de inventar un método
de asignación. Los métodos de reducción sí ajustan en el prefijo y transforman el sufijo.
Los detectores de anomalías calculan los scores de holdout sin reajustar el estimador.

## Estabilidad, trazabilidad e interpretación

Para clustering se pueden habilitar entre cero y cinco repeticiones de remuestreo. Se registra
el Adjusted Rand Index (ARI) entre las etiquetas del ajuste completo del prefijo y las obtenidas
en submuestras deterministas del mismo prefijo. El promedio es un diagnóstico de estabilidad,
no un umbral universal de calidad ni una prueba de que los clusters representen regímenes.
Los scores mayores de anomalía indican observaciones más inusuales respecto al ajuste entrenado;
la interpretación depende del algoritmo y de la distribución.

Los resultados guardan algoritmo, operación, parámetros resueltos, semilla, versión de
scikit-learn, variables, tamaño de prefijo/holdout, huella de entrenamiento, estadísticas de
escalado y limitaciones. Los objetos de estimador no se serializan dentro del contrato.

## Pruebas y criterios de aceptación

El workflow dedicado ejecuta Ruff, comprobación de formato, mypy, pruebas de las diez familias
de comportamiento, y regresiones seleccionadas de F28–F29. El CI general de la PR debe validar
las versiones Python 3.11 y 3.12 y ejecutar el conjunto completo más Security.

Las pruebas deben demostrar, como mínimo: registro explícito por operación; etiquetas de
clustering y estabilidad reproducibles; embeddings para ambos segmentos; scores y banderas de
anomalía finitos; ausencia de influencia del holdout sobre huella, escalado y ajuste del prefijo;
rechazo de entradas inválidas y presupuestos fuera de límites. El resultado no promueve modelos
a producción ni hace órdenes, backtests, selección económica o controles de riesgo.

## Limitaciones operativas

- El llamador es responsable de ordenar temporalmente las filas cuando el contexto lo requiere.
- El valor geométrico de un cluster no implica una categoría económica, régimen o causa.
- Las componentes de PCA, Kernel PCA e ICA no tienen interpretación económica automática.
- Los outliers son relativos al modelo, al escalado y a la distribución de entrenamiento.
- No se habilita ajuste automático de hiperparámetros, deep learning, visualización ni síntesis de señales.
