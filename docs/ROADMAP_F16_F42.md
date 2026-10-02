# C-MATH-AI — Roadmap técnico F16–F42

**Estado:** F19 IMPLEMENTADA  
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
**Objetivo:** volatilidad y procesos de reversión.  
**Alcance:** rolling volatility, EWMA, ARCH/GARCH, EGARCH, GJR-GARCH, stochastic volatility, DCC-GARCH, Ornstein-Uhlenbeck y mean reversion.  
**Cambios:** quant/volatility/ y/o quant/stochastic/, contratos, optimización y tests.  
**Reglas:** anualización explícita, restricciones de parámetros, convergencia comprobada.  
**Cierre:** modelos con validación numérica y supuestos documentados.

## F21 — Risk Engine
**Objetivo:** riesgo de posiciones y cartera.  
**Alcance:** beta, drawdown, Sharpe/Sortino/Treynor/Calmar/Omega/Information Ratio, exposición, concentración, leverage, liquidez y turnover.  
**Cambios:** risk/metrics/, risk/constraints/, integración con portfolio y CI.  
**Cierre:** evaluación tipada capaz de producir ALLOW/REDUCE/BLOCK.

## F22 — Tail Risk
**Objetivo:** pérdidas extremas y escenarios adversos.  
**Alcance:** historical/parametric/Monte Carlo VaR, CVaR/Expected Shortfall, stress testing y EVT cuando proceda.  
**Cambios:** risk/tail/ o equivalente, escenarios versionados, tests y CI.  
**Reglas:** confianza, horizonte, metodología y semillas explícitos.  
**Cierre:** métricas extremas reproducibles y consumibles por Risk Governor.

## F23 — Correlation / Dependency
**Objetivo:** dependencia entre activos y factores.  
**Alcance:** covarianza/correlación, rolling dependency, rank correlations, shrinkage, PCA/factores y copulas cuando proceda.  
**Reglas:** simetría, PSD cuando aplique, muestra mínima, ventanas y missing data explícitos.  
**Cierre:** matrices validadas para portfolio/risk/stat-arb.

## F24 — Cointegration / Statistical Arbitrage
**Objetivo:** relaciones de largo plazo y spreads.  
**Alcance:** ADF, KPSS, Phillips-Perron, Engle-Granger, Johansen, hedge ratio, spread, z-score, half-life y pairs statistics.  
**Reglas:** supuestos estadísticos explícitos, estimación sin futuro, distinguir correlación de cointegración y controlar selection/multiple testing.  
**Cierre:** spreads reproducibles y temporalmente válidos.

## F25 — Derivatives
**Objetivo:** pricing y Greeks.  
**Alcance:** Black-Scholes, binomial, trinomial, Monte Carlo, Greeks, implied volatility y superficies.  
**Cambios:** quant/derivatives/, contratos, tests analíticos/numerical.  
**Reglas:** tasas/dividendos/unidades explícitos, inputs acotados, estabilidad y calibración separada.  
**Cierre:** pricing/Greeks reproducibles y auditables.

## F26 — Regime Detection
**Objetivo:** detectar estados y transiciones de mercado.  
**Alcance:** HMM, Markov switching, change-point, CUSUM, Bayesian change detection y volatility regimes.  
**Reglas:** probabilidades cuando proceda, labels/versiones, estimación sin futuro, estabilidad por ventana.  
**Cierre:** output probabilístico/explicable integrable con ensemble.

## F27 — Classical ML
**Objetivo:** modelos ML sobre features validadas.  
**Alcance:** linear/logistic regression, Ridge, Lasso, Elastic Net, SVM/SVR, KNN, Decision Tree y Random Forest.  
**Cambios:** contratos dataset/model/prediction, fit/predict, metadata, evaluación, calibration y artefactos.  
**Reglas:** split temporal, preprocessing sólo sobre train, seeds y lineage.  
**Cierre:** modelos reproducibles y OOS.

## F28 — Gradient Boosting
**Objetivo:** boosting bajo el mismo contrato ML.  
**Alcance:** Gradient Boosting y adapters XGBoost/LightGBM/CatBoost si se aprueban dependencias/licencias; stacking/blending sólo con validación adecuada.  
**Reglas:** early stopping temporal, hiperparámetros versionados, feature importance no equivale a causalidad.  
**Cierre:** boosting reproducible e integrado.

## F29 — Unsupervised ML
**Objetivo:** clusters, factores y anomalías.  
**Alcance:** K-Means, hierarchical clustering, DBSCAN, GMM, PCA, Kernel PCA, ICA y anomaly detection.  
**Reglas:** scaling sin futuro, criterios/estabilidad registrados, no asumir que cluster = régimen.  
**Cierre:** outputs reproducibles y separados de interpretación económica automática.

## F30 — Deep Learning
**Objetivo:** modelos neuronales temporales/representacionales.  
**Alcance:** MLP, LSTM, GRU, TCN, autoencoders, VAE, attention, Transformer y TFT.  
**Reglas:** baselines clásicos previos, seeds, checkpoints, datasets/versiones reproducibles, early stopping temporal y OOS.  
**Cierre:** ningún modelo profundo entra al ensemble sin protocolo de validación.

## F31 — Feature Engineering
**Objetivo:** catálogo/pipeline unificado de features.  
**Alcance:** precio/retorno, volatilidad, volumen, fundamentales, macro, cross-sectional, régimen, riesgo, rolling/lagging e interacciones controladas.  
**Reglas:** cada feature tendrá definición, frecuencia, unidades, lookback, disponibilidad, versión y lineage; point-in-time safe.  
**Cierre:** feature catalog reproducible para ML/ensemble.

## F32 — Algorithm Registry
**Objetivo:** formalizar el catálogo de 300+ algoritmos.  
**Registro:** id, nombre, categoría, versión, inputs/outputs, parámetros, fórmula/método, unidades, frecuencia, requisitos, limitaciones y estado de tests.  
**Cambios:** quant/registry/, metadata schema, discovery y validación.  
**Cierre:** algoritmos descubribles sin imports manuales dispersos.

## F33 — Quant Ensemble
**Objetivo:** combinar evidencia heterogénea conservando contribuciones.  
**Alcance:** normalización, weighting, agreement/disagreement, score, uncertainty y attribution.  
**Reglas:** pesos versionados, evitar doble conteo de señales correlacionadas, conservar evidencia original y no saltar Risk Governor.  
**Cierre:** ensemble auditable y versionado.

## F34 — Meta-model / Decision Engine
**Objetivo:** producir el DecisionRecord.  
**Output:** BUY/HOLD/SELL, probabilidades, expected return/risk, horizon, confidence, agreement, evidence y constraints.  
**Reglas:** calibración, thresholds versionados, abstención/reject, Risk Governor posterior, LLM sólo como orquestación/interpretación.  
**Cierre:** decisión reproducible sin autoridad de ejecución.

## F35 — Portfolio Optimization
**Objetivo:** convertir señales elegibles en asignaciones.  
**Alcance:** equal weight, mean-variance, minimum variance, maximum Sharpe, risk parity/ERC, inverse volatility, HRP, Black-Litterman y maximum diversification.  
**Reglas:** restricciones explícitas, solver status, infeasibility, costes/turnover y concentración.  
**Cierre:** TargetAllocation validado.

## F36 — Position Sizing
**Objetivo:** determinar tamaño dentro de límites.  
**Alcance:** fixed fractional, volatility-based, risk-based, Kelly, fractional Kelly y confidence weighting.  
**Reglas:** inputs válidos, límites, liquidez y riesgo; tamaño cero permitido.  
**Cierre:** SizingResult auditable y subordinado a portfolio/risk.

## F37 — Portfolio Risk Governor
**Objetivo:** veto/reducción final antes de ejecución.  
**Controles:** posición, leverage, sector/factor, correlación, drawdown, daily loss, liquidez, tail risk, turnover y calidad de datos/modelos.  
**Output:** ALLOW/REDUCE/BLOCK + razones.  
**Regla:** ningún módulo puede anular un BLOCK.  
**Cierre:** Governor integrado con portfolio y execution mediante contratos.

## F38 — Backtesting
**Objetivo:** simulación histórica end-to-end.  
**Flujo:** Signal → Order → Execution → Slippage → Commission → Portfolio → PnL.  
**Cambios:** event loop, lifecycle de órdenes, fills, costes, accounting, métricas y audit trail.  
**Reglas:** corporate actions, información disponible, delistings cuando existan, costes/slippage/liquidez realistas.  
**Cierre:** backtest reproducible y contablemente consistente.

## F39 — Walk-Forward / OOS
**Objetivo:** validación temporal estricta.  
**Modos:** rolling, expanding, anchored.  
**Métricas:** retorno, volatilidad, drawdown, Sharpe/Sortino, turnover, costes y estabilidad por ventana/régimen.  
**Reglas:** ningún fit con test; parámetros congelados durante test; dispersión visible.  
**Cierre:** framework OOS reproducible.

## F40 — Monte Carlo + Adversarial Validation
**Objetivo:** medir fragilidad.  
**Alcance:** bootstrap/resampling, perturbación de parámetros, costes/slippage, execution delay, shocks de volatilidad/correlación y escenarios adversos.  
**Reglas:** semillas, número de simulaciones, distribuciones y supuestos versionados.  
**Cierre:** sensibilidad/fragilidad cuantificada, no sólo resultados favorables.

## F41 — Paper Trading + Execution
**Objetivo:** reproducir operación sin dinero real y preparar adapters futuros.  
**Cambios:** execution/paper/, orders, fills, execution policies, accounting y audit.  
**Reglas:** sólo decisiones risk-approved, idempotencia, estados explícitos, slippage/comisiones configurables, credenciales aisladas y sin órdenes directas desde LLM.  
**Cierre:** paper trading end-to-end.

## F42 — C-MATH-AI Platform 1.0
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
F26 → F27 → F28 → F29 → F30
             ↓
F31 → F32 → F33 → F34
             ↓
F35 → F36 → F37
             ↓
F38 → F39 → F40
             ↓
F41 → F42
```

La numeración orienta el roadmap; las dependencias contractuales tienen prioridad.

## Criterio de avance

Una fase no se considera lista sólo porque exista código. Debe tener implementación comprometida, contratos, tests relevantes, documentación, CI y ausencia de regresiones conocidas que afecten al alcance siguiente.

## Estado

- F01–F15: implementadas según el roadmap actual.
- F16–F19: implementadas según el alcance técnico y sus validaciones CI.
- F20–F42: alcance técnico documentado.
- **Siguiente implementación: F20.**
