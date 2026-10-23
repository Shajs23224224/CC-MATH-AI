# C-MATH-AI — Roadmap técnico F16–F43

**Estado:** F21–F35 integradas en `main`; F36 implementada en `feature/f36-portfolio-optimization`, pendiente de CI/revisión/merge.
**Base:** SYSTEM_SPEC.md, docs/ARCHITECTURE.md y contratos implementados hasta F18.

## Reglas globales
1. Matemática y finanzas críticas: deterministas, tipadas y testeables.
2. LLM/IA no sustituye cálculos, contabilidad, validaciones, límites de riesgo ni ejecución.
3. Prohibidos look-ahead, leakage y survivorship bias.
4. Cada algoritmo: contrato, metadatos, unidades, parámetros, limitaciones y tests.
5. Resultados de decisión: trazabilidad de datos, versiones, parámetros y configuración.
6. Cada fase material tendrá validación CI.
7. Reutilizar contratos existentes y respetar la dirección de dependencias.
8. Ningún backtest histórico se interpreta como garantía futura.

## F16 — Technical Analysis
**Objetivo:** indicadores deterministas sobre OHLCV.  
**Alcance:** SMA, EMA, WMA, momentum, ROC, RSI, MACD, ATR, Bollinger, stochastic, Williams %R, ADX/DMI, Donchian, VWAP y volumen.  
**Cambios:** quant/technical/, contratos sólo cuando sean necesarios, tests numéricos, documentación y CI.  
**Invariantes:** ventanas válidas, warm-up explícito, timestamps preservados, sin futuro.  
**Cierre:** indicadores comprometidos con referencias numéricas, casos de borde y no-lookahead.

## F17 — Fundamental Analysis
**Objetivo:** ratios, márgenes, crecimiento, calidad y solvencia point-in-time.  
**Alcance:** ROE, ROA, ROIC, ROCE, márgenes, FCF, cash conversion, payout/retention, leverage, interest coverage y crecimiento de revenue/EBITDA/EPS/FCF.  
**Cambios:** quant/fundamental/, contratos, tests, docs y CI.  
**Reglas:** usar reported_at, denominadores inválidos explícitos, no imputación silenciosa.  
**Cierre:** métricas reproducibles y point-in-time safe.

## F18 — Valuation
**Objetivo:** valoración fundamental y relativa.  
**Alcance:** P/E, P/S, P/B, EV/EBITDA, EV/FCF, earnings yield, DDM, Gordon, DCF, multi-stage DCF, residual income, NAV, SOTP, reverse DCF y comparables.  
**Cambios:** quant/valuation/, inputs/outputs tipados, sensibilidad, tests y CI.  
**Reglas:** moneda/unidades explícitas, supuestos versionados, evitar doble conteo de deuda/caja.  
**Cierre:** valoración reproducible, auditable y con casos extremos.

## F19 — Time Series
**Objetivo:** econometría temporal.  
**Alcance:** AR, MA, ARMA, ARIMA, SARIMA, ARIMAX, VAR, VECM, exponential smoothing, Holt y Holt-Winters.  
**Cambios:** quant/time_series/, contratos, diagnóstico de residuos, tests y CI.  
**Reglas:** separación temporal, parámetros sin futuro, estacionariedad/orden y diagnóstico documentados.  
**Cierre:** modelos base validados temporalmente y sin leakage. **IMPLEMENTADO.**

## F20 — Volatility & Stochastic
**Estado:** IMPLEMENTADA; corrección de formato incluida para desbloquear la validación automatizada
**Objetivo:** volatilidad y procesos de reversión.  
**Alcance:** rolling volatility, EWMA, ARCH/GARCH, EGARCH, GJR-GARCH, stochastic volatility, DCC-GARCH, Ornstein-Uhlenbeck y mean reversion.  
**Cambios:** quant/volatility/ y/o quant/stochastic/, contratos, optimización y tests.  
**Reglas:** anualización explícita, restricciones de parámetros, convergencia comprobada.  
**Cierre:** modelos con validación numérica y supuestos documentados.

## F21 — Risk Engine
**Estado:** IMPLEMENTADA; PR #6 fusionada en main; CI/Security verificados
**Objetivo:** riesgo de posiciones y cartera.  
**Alcance:** beta, drawdown, Sharpe/Sortino/Treynor/Calmar/Omega/Information Ratio, exposición, concentración, leverage, liquidez y turnover.  
**Cambios:** risk/metrics/, risk/constraints/, integración con portfolio y CI.  
**Cierre:** evaluación tipada capaz de producir ALLOW/REDUCE/BLOCK, con métricas auditables y fallo seguro ante métricas requeridas ausentes.

## F22 — Tail Risk
**Estado:** IMPLEMENTADA; PR #7 fusionada en main; CI general, F21 y Security aprobados
**Objetivo:** pérdidas extremas y escenarios adversos.  
**Alcance:** historical/parametric/Monte Carlo VaR, CVaR/Expected Shortfall, stress testing y EVT cuando proceda.  
**Cambios:** risk/tail/ o equivalente, escenarios versionados, tests y CI.  
**Reglas:** confianza, horizonte, metodología y semillas explícitos.  
**Cierre:** métricas extremas reproducibles y consumibles por Risk Governor.

## F23 — Correlation / Dependency
**Estado:** IMPLEMENTADA; CI F23, CI general y Security aprobados
**Objetivo:** dependencia entre activos y factores.  
**Alcance:** covarianza/correlación, rolling dependency, rank correlations, shrinkage, PCA/factores y copulas cuando proceda.  
**Reglas:** simetría, PSD cuando aplique, muestra mínima, ventanas y missing data explícitos.  
**Cierre:** matrices validadas para portfolio/risk/stat-arb.

## F24 — Cointegration / Statistical Arbitrage
**Estado:** IMPLEMENTADA; PR #9 fusionada en main; CI F24, CI general Python 3.11/3.12 y Security aprobados
**Objetivo:** relaciones de largo plazo y spreads.  
**Alcance:** ADF, KPSS, Phillips-Perron, Engle-Granger, Johansen, hedge ratio, spread, z-score, half-life y pairs statistics.  
**Reglas:** supuestos estadísticos explícitos, estimación sin futuro, distinguir correlación de cointegración y controlar selection/multiple testing.  
**Cierre:** spreads reproducibles y temporalmente válidos.

## F25 — Derivatives
**Estado:** IMPLEMENTADA; PR #10 fusionada en main; F25, CI general y Security aprobados
**Objetivo:** pricing y Greeks.  
**Alcance:** Black-Scholes, binomial, trinomial, Monte Carlo, Greeks, implied volatility y superficies.  
**Cambios:** quant/derivatives/, contratos, tests analíticos/numerical.  
**Reglas:** tasas/dividendos/unidades explícitos, inputs acotados, estabilidad y calibración separada.  
**Cierre:** pricing/Greeks reproducibles y auditables.

## F26 — Regime Detection
**Estado:** INTEGRADA en `main`; PR #11 fusionada; workflows de validación y regresiones aprobados.
**Objetivo:** detectar estados y transiciones de mercado.  
**Alcance:** HMM, Markov switching, change-point, CUSUM, Bayesian change detection y volatility regimes.  
**Reglas:** probabilidades cuando proceda, labels/versiones, estimación sin futuro, estabilidad por ventana.  
**Cierre:** output probabilístico/explicable integrable con ensemble.

## F27 — Probabilistic Forecasting Engine
**Estado:** INTEGRADA y endurecida en `main`; PR #12 más PR #13 para causalidad estricta en walk-forward; CI y Security aprobados.
**Objetivo:** estimar distribuciones predictivas de retornos futuros con incertidumbre explícita, no sólo una predicción puntual.  
**Alcance:** bootstrap empírico, simulación de paths, cuantiles e intervalos centrales (50/80/90/95%), probabilidad de pérdida, bootstrap condicionado a regímenes de F26, calibración de cobertura/Brier/Winkler y evaluación walk-forward expansiva.  
**Reglas:** retornos simples compuestos, semillas explícitas, entrenamiento anterior a cada origen, labels/transiciones de régimen causales y calibradas en train; documentar que IID bootstrap no conserva autocorrelación. No emite órdenes ni sustituye F21 Risk Engine.  
**Cierre:** contratos reproducibles, pruebas numéricas y de leakage, calibración OOS, documentación y CI propia con regresiones F26 y F21–F25.

## F28 — Classical ML and adaptive mathematical discovery
**Base integrada:** PR #14. La ampliación de inteligencia matemática añade coordinación adaptativa, generación evolutiva acotada de expresiones y evaluación independiente, sin duplicar los adaptadores ML o la memoria de experimentos existentes.
**Objetivo:** elegir de forma reproducible entre estrategias matemáticas candidatas, descubrir regularidades verificables en datos y conservar evidencia auditable.
**Implementado:** contratos tipados de árbol de expresiones, configuración, candidatos y reportes; comparador lineal/polinómico/evolutivo; generación mediante selección, mutación y cruce de subárboles; gramática autorizada con aritmética, división con rechazo de dominio inválido, sin/cos/tanh/abs; calibración de constantes sobre train; validación cronológica para selección y test reservado para evaluación final; asociaciones calculadas solo en el prefijo de entrenamiento; verificación numérica diferencial y persistencia JSONL opcional.
**Reglas:** ningún candidato ejecuta código Python; restricciones de profundidad, nodos, términos, observaciones y evaluaciones; sin fuga temporal; correlación no implica causalidad; los candidatos siguen en estado `EXPERIMENTAL`; ausencia de contraejemplos no es demostración formal.
**Límites explícitos:** regresión supervisada para el coordinador adaptativo; no implementa aún síntesis general de programas, búsqueda Pareto multiobjetivo, descubrimiento causal, búsqueda automática general de hiperparámetros, calibración probabilística, clustering, deep learning, prueba formal ni interfaz gráfica. La gramática evolutiva es deliberadamente finita.
**Cierre:** verificar el workflow F28 específico, CI en Python 3.11/3.12, regresiones F21–F27, Security y pruebas sintéticas de comportamiento antes de promocionar cambios.

## F29 — Gradient Boosting
**Estado:** Integrada mediante PR #17; workflow F29, CI general (Python 3.11/3.12) y Security aprobados.  
**Objetivo:** integrar boosting supervisado en el contrato y registro explícito de F28.  
**Implementado en esta fase:** GradientBoostingRegressor y GradientBoostingClassifier; selección temporal del número de árboles con validación interna cronológica; refit sobre el prefijo completo de entrenamiento; métricas OOS en el test final reservado; metadatos del árbol seleccionado, presupuesto, validación, semilla y fingerprint.  
**Reglas:** no añade XGBoost/LightGBM/CatBoost sin revisión separada de dependencias y licencias; early stopping no observa el test final; presupuesto máximo de árboles acotado; feature importance no implica causalidad; sin stacking/blending hasta disponer de validación out-of-fold explícita.  
**Cierre:** workflow F29, lint, format, mypy, pruebas de comportamiento y regresiones F21–F28, CI general y Security aprobados.

## F30 — Unsupervised ML
**Objetivo:** clusters, factores y anomalías.  
**Alcance:** K-Means, hierarchical clustering, DBSCAN, GMM, PCA, Kernel PCA, ICA y anomaly detection.  
**Reglas:** scaling sin futuro, criterios/estabilidad registrados, no asumir que cluster = régimen.  
**Cierre:** outputs reproducibles y separados de interpretación económica automática.

## F31 — Deep Learning
**Objetivo:** modelos neuronales temporales/representacionales.  
**Alcance:** MLP, LSTM, GRU, TCN, autoencoders, VAE, attention, Transformer y TFT.  
**Reglas:** baselines clásicos previos, seeds, checkpoints, datasets/versiones reproducibles, early stopping temporal y OOS.  
**Cierre:** ningún modelo profundo entra al ensemble sin protocolo de validación.

## F32 — Feature Engineering
**Objetivo:** catálogo/pipeline unificado de features.  
**Alcance:** precio/retorno, volatilidad, volumen, fundamentales, macro, cross-sectional, régimen, riesgo, rolling/lagging e interacciones controladas.  
**Reglas:** cada feature tendrá definición, frecuencia, unidades, lookback, disponibilidad, versión y lineage; point-in-time safe.  
**Cierre:** feature catalog reproducible para ML/ensemble.

## F33 — Algorithm Registry
**Estado:** integrada en `main` mediante PR #21 (merge commit `4f400ff`); CI general, Python 3.11/3.12 y Security aprobados.  
**Objetivo:** establecer la infraestructura del registro versionado y un catálogo de al menos 300 entradas con metadatos tipados. La implementación contiene 304 IDs únicos; este conteo valida la integridad del catálogo, no la corrección funcional de cada algoritmo.  
**Implementado:** contratos inmutables para metadatos, puertos y parámetros; IDs/versiones validados; registro versionado con inserción por lotes atómica; descubrimiento por categoría, lifecycle, puertos y tags; compatibilidad por nombre/tipo/unidad; estados separados de ciclo de vida y evidencia de tests; manifiesto y workflow de validación.  
**Corrección final de evidencia:** las entradas semilla con referencias declaradas a APIs existentes conservan `implementation_ref`; las entradas de planificación exponen `planned_implementation_ref`, permanecen `experimental` y no simulan tener una implementación disponible. Los 304 elementos continúan con `test_status=not_run` para evidencia funcional individual.  
**Seguridad:** el registro sólo gestiona metadatos; no importa ni ejecuta referencias, ni promueve automáticamente algoritmos a `validated`.  
**Cierre de F33:** infraestructura del registro y metadatos cerrados; validar individualmente las implementaciones a medida que existan queda como trabajo de calidad de cada algoritmo, no como una afirmación implícita del catálogo.

## F34 — Quant Ensemble
**Estado:** integrada en `main` mediante [PR #22](https://github.com/Shajs23224224/CC-MATH-AI/pull/22), merge commit `26e7888`. En el commit de la PR `41aead9` pasaron el workflow F34 (Python 3.11/3.12), el workflow F33, CI general y Security.
**Objetivo:** combinar evidencia heterogénea con un resultado reproducible sin perder contribuciones ni exclusiones.
**Implementado:** contratos inmutables de evidencia, solicitud, configuración, contribución y resultado; normalización acotada por `bounded`, `tanh` o `zscore`; pesos predeterminados y por fuente versionados; ajuste por confianza y fiabilidad; límite de peso total por grupo de correlación; score, acuerdo, incertidumbre, abstención, atribución por fuente y fingerprint SHA-256 de entradas/configuración.
**Reglas:** timestamps con zona horaria; rechazar evidencia posterior a `as_of`; un único horizonte y fuente por solicitud; excluir y auditar fuentes no elegibles o de peso efectivo cero; la cantidad mínima de grupos independientes es una condición de abstención. El cap por grupo es una heurística de ingeniería, no un estimador estadístico de correlación.
**Límites:** no aprende pesos, no calibra probabilidades, no produce BUY/HOLD/SELL, no optimiza cartera, no ordena operaciones ni puede anular Risk Governor. F35 consume la evidencia agregada.
**Cierre:** contratos tipados, pruebas numéricas/temporales, trazabilidad de contribuciones, regresiones F29–F33, CI Python 3.11/3.12, Security y documentación aprobados.

## F35 — Meta-model / Decision Engine
**Estado:** completada e integrada en `main` mediante PR #24; CI y Security aprobados.
**Objetivo:** convertir evidencia F34 y un Forecast de fuente/version explícitas al mismo horizonte en un DecisionRecord reproducible.
**Implementado:** contratos tipados e inmutables para probabilidades BUY/HOLD/SELL, solicitud, restricciones, contribuciones de evidencia y salida; umbrales/configuración semánticamente versionados; softmax estable y metadatos de calibración; gates de acuerdo, incertidumbre, forecast-direction alignment y constraints bloqueantes; abstención segura con placeholder HOLD no accionable; fingerprint SHA-256 canónico; position size fijo en cero para mantener sizing downstream.
**Calibración:** grid search determinista para temperatura y bias HOLD con log-loss multicategoría, ventana cronológica de ajuste y validación posterior separada; purga de observaciones de train cuyos labels aún no estaban disponibles al inicio de validación; fingerprint/dataset/window/loss provenance; exige cobertura de clases y al menos 30 muestras por defecto. No se marca calibración válida si empeora el baseline de validación.
**Reglas:** no aceptar forecast posterior a `as_of`, horizontes mezclados ni timestamps sin zona horaria; por defecto las probabilidades no calibradas fuerzan abstención; ningún output autoriza trades, crea tamaño de posición ni anula Risk Governor. LLM sólo para orquestación/interpretación.
**Límites:** el motor no prueba la veracidad de labels/snapshots externos, ni garantiza calibración económica, rentabilidad o rendimiento en vivo; las pruebas sintéticas son validación de ingeniería. F36/F37/F38 mantienen cartera, tamaño y control de riesgo posteriores.
**Cierre:** completado. Workflow F35 pasó en Python 3.11/3.12 (Ruff, formato, mypy, pruebas unitarias y regresiones); CI general y Security pasaron, junto con los workflows de regresión que se activaron para F22–F34.

## F36 — Portfolio Optimization
**Estado:** implementada en `feature/f36-portfolio-optimization`; aceptación final pendiente de workflow F36, CI general, Security y PR.
**Objetivo:** convertir outputs tipados de F35 en pesos objetivo long-only y fully-invested mediante una matriz F23 de covarianza con timestamp/horizonte y procedencia explícitos.
**Métodos:** equal weight, mean-variance, minimum variance, maximum Sharpe, risk parity/ERC, inverse volatility, HRP, Black-Litterman con vistas/confianzas explícitas y maximum diversification.
**Restricciones:** bounds por activo, exclusión de abstentions/SELL, HOLD sólo preserva o reduce posiciones existentes, límites sectoriales, turnover máximo, concentración HHI y suma de pesos a 1. Evalúa costes estimados, reporta estado SLSQP, iteraciones, diagnóstico medido por restricción, ridge de covarianza y huellas SHA-256.
**Reglas temporales:** no aceptar timestamps futuros, horizontes incompatibles, covariance matrix de correlación en lugar de covariance, entradas no finitas ni dimensiones desalineadas. No anualiza retornos/covarianza implícitamente.
**Fallo seguro:** constraints/bounds contradictorios generan estado INFEASIBLE cuando se detecta; solver incapaz de encontrar una candidata válida produce FAILED; ambos devuelven pesos cero para evitar consumo accidental. Solo OPTIMAL/FEASIBLE con todas las restricciones aprobadas son TargetAllocation utilizables.
**Límites:** solo long-only y sin objetivo explícito de cash; costes estimados, no cotizaciones de ejecución; F36 no sustituye Risk Governor, F37 sizing, backtest/walk-forward ni paper trading. Métodos y pruebas sintéticas validan ingeniería, no rentabilidad.
**Cierre:** workflow F36 Python 3.11/3.12, Ruff/format, mypy, pruebas de nueve métodos/constraints/leakage y regresiones F33–F35, F23/F21, CI general y Security aprobados.

## F37 — Position Sizing
**Objetivo:** determinar tamaño dentro de límites.  
**Alcance:** fixed fractional, volatility-based, risk-based, Kelly, fractional Kelly y confidence weighting.  
**Reglas:** inputs válidos, límites, liquidez y riesgo; tamaño cero permitido.  
**Cierre:** SizingResult auditable y subordinado a portfolio/risk.

## F38 — Portfolio Risk Governor
**Objetivo:** veto/reducción final antes de ejecución.  
**Controles:** posición, leverage, sector/factor, correlación, drawdown, daily loss, liquidez, tail risk, turnover y calidad de datos/modelos.  
**Output:** ALLOW/REDUCE/BLOCK + razones.  
**Regla:** ningún módulo puede anular un BLOCK.  
**Cierre:** Governor integrado con portfolio y execution mediante contratos.

## F39 — Backtesting
**Objetivo:** simulación histórica end-to-end.  
**Flujo:** Signal → Order → Execution → Slippage → Commission → Portfolio → PnL.  
**Cambios:** event loop, lifecycle de órdenes, fills, costes, accounting, métricas y audit trail.  
**Reglas:** corporate actions, información disponible, delistings cuando existan, costes/slippage/liquidez realistas.  
**Cierre:** backtest reproducible y contablemente consistente.

## F40 — Walk-Forward / OOS
**Objetivo:** validación temporal estricta.  
**Modos:** rolling, expanding, anchored.  
**Métricas:** retorno, volatilidad, drawdown, Sharpe/Sortino, turnover, costes y estabilidad por ventana/régimen.  
**Reglas:** ningún fit con test; parámetros congelados durante test; dispersión visible.  
**Cierre:** framework OOS reproducible.

## F41 — Monte Carlo + Adversarial Validation
**Objetivo:** medir fragilidad.  
**Alcance:** bootstrap/resampling, perturbación de parámetros, costes/slippage, execution delay, shocks de volatilidad/correlación y escenarios adversos.  
**Reglas:** semillas, número de simulaciones, distribuciones y supuestos versionados.  
**Cierre:** sensibilidad/fragilidad cuantificada, no sólo resultados favorables.

## F42 — Paper Trading + Execution
**Objetivo:** reproducir operación sin dinero real y preparar adapters futuros.  
**Cambios:** execution/paper/, orders, fills, execution policies, accounting y audit.  
**Reglas:** sólo decisiones risk-approved, idempotencia, estados explícitos, slippage/comisiones configurables, credenciales aisladas y sin órdenes directas desde LLM.  
**Cierre:** paper trading end-to-end.

## F43 — C-MATH-AI Platform 1.0
**Objetivo:** integración completa y operable.  
**Cadena:** DATA → QUALITY → FEATURES → 300+ ALGORITHMS → REGIMES → ML → ENSEMBLE → DECISION → PORTFOLIO → RISK → BACKTEST/OOS → PAPER → EXECUTION ADAPTER.  
**Cambios:** API/CLI/jobs, dashboard si procede, stores, registry, observabilidad, auditoría y deployment.  
**Release:** contratos estables, CI/security, reproducibilidad, documentación, smoke/integration tests, rollback y manejo de fallos.  
**No implica:** trading real automático, garantías de rentabilidad ni bypass del Risk Governor.  
**Cierre:** cadena end-to-end reproducible desde datos hasta paper trading.

## Dependencias
```
F16 → F17 → F18 → F19 → F20
             ↓
F21 → F22 → F23 → F24 → F25
             ↓
F26 → F27 → F28 → F29 → F30 → F31
                         ↓
F32 → F33 → F34 → F35
             ↓
F36 → F37 → F38
             ↓
F39 → F40 → F41
             ↓
F42 → F43
```

La numeración orienta el roadmap; las dependencias contractuales tienen prioridad.

## Criterio de avance

Una fase no se considera lista sólo porque exista código. Debe tener implementación comprometida, contratos, tests relevantes, documentación, CI y ausencia de regresiones conocidas que afecten al alcance siguiente.

## Estado

- F01–F15: implementadas según el roadmap actual.
- F16–F18: implementadas según el alcance técnico y validadas por CI.
- F19: implementación completa; F19 Time Series, CI general y Security verificados correctamente.
- F20: implementación y pruebas específicas incorporadas; la corrección de formato de Ruff se revalida en la rama F21.
- F21: Risk Engine integrado en main mediante PR #6; CI general, Security y validación de fase aprobadas.
- F22: VaR histórico/paramétrico/Monte Carlo, CVaR, EVT y stress testing integrados en main mediante PR #7; CI general Python 3.11/3.12, Security, tests F22 y regresión F21 aprobados.
- F23: matrices Pearson/Spearman/Kendall, covarianza, OAS shrinkage, dependencia rolling y PCA integradas en main mediante PR #8; Ruff, formato, mypy, tests numéricos, regresiones F21/F22, CI Python 3.11/3.12 y Security aprobados.
- F24: ADF/KPSS/Phillips–Perron, Engle–Granger, Johansen, pares cointegrados, spread/z-score, half-life y Benjamini–Hochberg integrados en main mediante PR #9; CI F24, Ruff, mypy, pruebas numéricas, regresiones F21/F22/F23, CI general Python 3.11/3.12 y Security aprobados.
- F25: Black–Scholes, árboles binomiales/trinomiales, Monte Carlo, Greeks, volatilidad implícita y superficie de volatilidad integrados en main mediante PR #10; F25, CI general y Security aprobados.
- F26: HMM/Markov switching, CUSUM, Bayesian online change points y regímenes de volatilidad integrados en `main` mediante PR #11 (`96a5646`); F26/general CI/Security y regresiones aprobados.
- F27: forecasting probabilístico integrado y endurecido en `main` mediante PR #12 y PR #13 (`b07b3c4`); CI Python 3.11/3.12 y Security aprobados.
- F28: ML clásico, patrones exploratorios, regresión simbólica acotada y motor adaptativo con búsqueda evolutiva de árboles de expresión; el estado de integración se determina por PR, CI y Security, no por la presencia de documentación.
- F29: integrada en `main` por PR #17 (`96fcdb5`); ver [diseño técnico](GRADIENT_BOOSTING.md).
- F30: integrada en `main` mediante [PR #18](https://github.com/Shajs23224224/CC-MATH-AI/pull/18), merge commit `25de38d`; workflow F30, CI general, regresiones de las fases anteriores y Security completados con éxito en el HEAD final antes de la fusión.
- F31: integrada en `main` mediante [PR #19](https://github.com/Shajs23224224/CC-MATH-AI/pull/19); ver [diseño técnico](DEEP_LEARNING.md).
- F32: integrada en `main` mediante [PR #20](https://github.com/Shajs23224224/CC-MATH-AI/pull/20); ver [diseño técnico](FEATURE_ENGINEERING.md).
- F33: integrada en `main` mediante [PR #21](https://github.com/Shajs23224224/CC-MATH-AI/pull/21); las referencias verificadas y los objetivos de implementación planificados están diferenciados.
- F34: integrada en `main` mediante [PR #22](https://github.com/Shajs23224224/CC-MATH-AI/pull/22), merge commit `26e7888`; workflow F34, workflow F33, CI general y Security pasaron en el head de la PR antes de fusionar.
- F35–F43: alcance documentado; avanzar por dependencias contractuales y gates de validación.
- **Siguiente fase planificada: F35 — Meta-model / Decision Engine.**
