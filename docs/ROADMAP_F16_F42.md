# C-MATH-AI — Roadmap técnico F16–F43

**Estado:** F21–F38 integradas en main; F39/PR #28, F40/PR #29 y F41/feature/f41-monte-carlo-adversarial son cambios apilados pendientes de su orden de validación e integración.
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
**Estado:** completada e integrada en `main` mediante PR #25 (merge commit `b7a2628`); workflow F36, F33, CI general, Security y regresiones activadas aprobados.
**Objetivo:** convertir outputs tipados de F35 en pesos objetivo long-only y fully-invested mediante una matriz F23 de covarianza con timestamp/horizonte y procedencia explícitos.
**Métodos:** equal weight, mean-variance, minimum variance, maximum Sharpe, risk parity/ERC, inverse volatility, HRP, Black-Litterman con vistas/confianzas explícitas y maximum diversification.
**Restricciones:** bounds por activo, exclusión de abstentions/SELL, HOLD sólo preserva o reduce posiciones existentes, límites sectoriales, turnover máximo, concentración HHI y suma de pesos a 1. Evalúa costes estimados, reporta estado SLSQP, iteraciones, diagnóstico medido por restricción, ridge de covarianza y huellas SHA-256.
**Reglas temporales:** no aceptar timestamps futuros, horizontes incompatibles, covariance matrix de correlación en lugar de covariance, entradas no finitas ni dimensiones desalineadas. No anualiza retornos/covarianza implícitamente.
**Fallo seguro:** constraints/bounds contradictorios generan estado INFEASIBLE cuando se detecta; solver incapaz de encontrar una candidata válida produce FAILED; ambos devuelven pesos cero para evitar consumo accidental. Solo OPTIMAL/FEASIBLE con todas las restricciones aprobadas son TargetAllocation utilizables.
**Límites:** solo long-only y sin objetivo explícito de cash; costes estimados, no cotizaciones de ejecución; F36 no sustituye Risk Governor, F37 sizing, backtest/walk-forward ni paper trading. Métodos y pruebas sintéticas validan ingeniería, no rentabilidad.
**Cierre:** completado. F36 pasó en Python 3.11/3.12 (Ruff, formato, mypy, pruebas de nueve métodos y restricciones); F33, CI general, Security y los workflows regresivos activados pasaron. Las pruebas sintéticas no acreditan rentabilidad ni idoneidad financiera.

## F37 — Position Sizing
**Estado:** integrada en `main` mediante [PR #26](https://github.com/Shajs23224224/CC-MATH-AI/pull/26), merge commit `f65ee22`.
**Objetivo:** convertir pesos aprobados por F36 en cantidades objetivo de forma determinista y auditable.
**Métodos:** fixed fractional, volatility-based, risk-based mediante presupuesto de pérdida a stop, Kelly, fractional Kelly y confidence weighting.
**Reglas:** el peso F36 es un techo inalterable; control por peso máximo, lote y participación en volumen medio diario; tamaños cero permitidos; residual en efectivo sin renormalización; timestamps/horizonte/símbolos alineados; fingerprints SHA-256.
**Límites:** no emite órdenes, no garantiza ejecuciones ni rentabilidad; no sustituye Risk Governor/F38. La liquidez es una estimación suministrada y las pruebas sintéticas sólo validan ingeniería.
**Validación:** workflow F37, CI general, Security y regresiones upstream pasaron en Python 3.11/3.12.

## F38 — Portfolio Risk Governor
**Estado:** integrada en `main` mediante [PR #27](https://github.com/Shajs23224224/CC-MATH-AI/pull/27), merge commit `f315c6e`.
**Objetivo:** veto/reducción determinista final después de F37 y antes de cualquier adaptador de ejecución.
**Controles:** peso por posición, gross leverage, HHI, sector/factor, correlación F23, drawdown, pérdida diaria, liquidez/ADV, CVaR F22, turnover, calidad de datos/modelos y frescura temporal.
**Output:** ALLOW/REDUCE/BLOCK, razones, métricas, restricciones activadas, cantidades por lote aprobadas y huellas SHA-256.
**Regla:** ausencia, antigüedad, futuro o unidades incorrectas en evidencia obligatoria bloquean; un BLOCK nunca puede convertirse en ALLOW; REDUCE redondea cantidades hacia abajo y puede acabar en BLOCK si no quedan lotes válidos.
**Límites:** no envía órdenes ni modela todavía derivados, cortos, FX o ejecución real. Parámetros predeterminados son configurables y no recomendaciones universales.
**Validación:** workflow F38, CI general, Security y regresiones upstream pasaron en Python 3.11/3.12.

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
**Estado:** implementación inicial en feature/f41-monte-carlo-adversarial, apilada sobre F40/PR #29 y F39/PR #28. Workflow F41 aprobado en Python 3.11/3.12 (14 tests por versión, Ruff, formato y mypy); workflows F39 y regresiones F22–F28 aprobados en HEAD 3fed645. CI general, Security y revisión final pendientes hasta integrar las dependencias y retargetear la PR a main.
**Objetivo:** cuantificar sensibilidad y fragilidad de resultados de backtest, sin afirmar capacidad predictiva futura.
**Implementado:** contratos versionables; bootstrap de retornos de equity mediante IID o moving-block con seed local, número de simulaciones, confianza y block size explícitos; distribución de retorno terminal, pérdida, expected shortfall y drawdown; escenarios de volatilidad, shock persistente de precio, comisión/slippage incrementales y latencia de señales que re-ejecutan F39; métricas de degradación, score heurístico de fragilidad y fingerprints SHA-256; estrés de matrices de correlación desde paneles multiactivo alineados.
**Reglas:** equity estrictamente positiva para bootstrap; series de correlación finitas, alineadas y no constantes; escenarios con nombres únicos y supuestos explícitos; ejecuciones retrasadas fuera del periodo disponible se descartan, nunca se inventan fills. Seeds, supuestos, entradas y resultados se incluyen en la huella.
**Límite importante:** el motor F39 admite un solo flujo OHLC. El estrés de correlación devuelve matrices diagnósticas y no atribuye PnL multiactivo; para eso se requiere un contrato posterior de contabilidad de cartera multiactivo. El bootstrap re-muestrea resultados observados y no elimina sesgo de selección ni look-ahead.
**Validación pendiente:** workflow F41 en Python 3.11/3.12, tests específicos, Ruff, mypy, CI general, Security y regresiones upstream sobre la rama final de la cadena. Los tests sintéticos validan ingeniería, no rentabilidad.


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
- F35: implementada en main mediante PR #24.
- F36: integrada en main mediante PR #25.
- F37: integrada en `main` mediante [PR #26](https://github.com/Shajs23224224/CC-MATH-AI/pull/26), merge commit `f65ee22`; workflow F37, CI general, Security y regresiones upstream aprobados.
- F38: integrada en `main` mediante [PR #27](https://github.com/Shajs23224224/CC-MATH-AI/pull/27), merge commit `f315c6e`; workflow F38, CI general, Security y regresiones upstream aprobados.
- F39: implementada en feature/f39-backtesting-engine; workflow F39, CI general en Python 3.11/3.12, Security y regresiones upstream aprobados en el HEAD validado; PR #28 sigue abierta.
- F40: implementación en feature/f40-walk-forward-oos, apilada sobre la rama F39; split modes, runner causal, métricas OOS, documentación y workflow específico añadidos; requiere CI final antes de cierre.
- F41: implementación inicial de Monte Carlo, perturbaciones adversariales, estrés de correlación, tests y workflow en feature/f41-monte-carlo-adversarial; workflow específico y regresiones F39/F22–F28 aprobados en HEAD 3fed645; CI general, Security y revisión final pendientes por la dependencia de F39/F40.
- F42–F43: alcance documentado; dependen de completar e integrar F39–F41.
- **Siguiente fase: F41 — Monte Carlo + Adversarial Validation.**


## F39 — Backtesting Engine
**Estado:** Implementación inicial en `feature/f39-backtesting-engine`; pendiente de CI y revisión.
**Objetivo:** simulación determinista de señales con ejecución en la apertura de la siguiente barra, contabilidad de posiciones, comisiones, slippage y métricas de cartera.
**Alcance inicial:** OHLC validado, timestamps estrictamente crecientes, posiciones enteras con signo, fills market-style next-open, curva de equity, retorno total, drawdown máximo, Sharpe, Sortino, turnover y pruebas temporales/contables.
**Seguridad temporal:** las señales del cierre no se ejecutan hasta la siguiente apertura; la señal de la última barra queda sin ejecutar; no se envían órdenes reales.
**Limitaciones:** sin fills parciales, órdenes limit/stop, impacto de mercado, financiación, corporate actions ni integración automática F35–F38. Los tests sintéticos validan ingeniería, no rentabilidad de estrategias.
**Validación:** workflow F39 y CI general en Python 3.11/3.12, Security y regresiones upstream aprobados en el HEAD `73c4e68`; PR #28 abierta y sin fusionar.

## F40 — Walk-Forward / OOS
**Estado:** implementación en `feature/f40-walk-forward-oos`, apilada sobre F39; pendiente de CI y revisión.
**Objetivo:** validación temporal estricta mediante ventanas OOS reproducibles.
**Modos:** rolling (train de longitud fija), expanding (prefijo creciente desde el primer registro) y anchored (train creciente desde un índice de anclaje explícito).
**Implementado:** contratos de configuración/splits, gap de purga opcional, ventanas de test no solapadas, ajuste por fold sólo sobre filas de entrenamiento, predicción sobre una vista temporal causal de sólo lectura, evaluación de cada ventana mediante F39, volatilidad anualizada, retorno/dispersión entre ventanas, drawdown, Sharpe/Sortino, turnover, costes, métricas por régimen opcionales, fingerprint SHA-256, tests y documentación.
**Reglas y límites:** el predictor ve como máximo el prefijo hasta la barra actual; las señales de cierre se ejecutan en la siguiente apertura; cada ventana comienza con capital en efectivo y posición plana; la señal final no se ejecuta dentro de la ventana. Las callbacks aún son código confiable: no se puede probar automáticamente que no usen estado global, labels futuros o datos externos con leakage. Las etiquetas de régimen las suministra la aplicación.
**Cierre requerido:** workflow F40 en Python 3.11/3.12, Ruff, mypy, CI general, Security y regresiones upstream aprobados antes de fusionar.
