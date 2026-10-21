# C-MATH-AI — Roadmap técnico F01–F15

**Estado de referencia:** fases iniciales implementadas según el README y el historial del repositorio. Este registro describe el alcance y el criterio de cierre; no sustituye la verificación actual de CI ni implica que cada workflow esté verde hoy.

**Roadmap completo:** [F16–F42](ROADMAP_F16_F42.md). Las reglas globales de diseño, arquitectura y seguridad están en [SYSTEM_SPEC.md](../SYSTEM_SPEC.md) y [ARCHITECTURE.md](ARCHITECTURE.md).

## Criterios globales para todas las fases

- Cambios pequeños, revisables y comprometidos en Git.
- Contratos explícitos, validación de entradas, errores definidos y unidades/frecuencias claras cuando aplique.
- Tests unitarios y numéricos; integración cuando haya límites entre módulos; regresiones para defectos corregidos.
- Sin look-ahead, leakage ni survivorship bias en datos, features o evaluación.
- Matemática financiera determinista: la IA no sustituye los cálculos ni puede saltarse controles de riesgo.
- Documentación actualizada y CI/Security verificados antes de declarar una fase cerrada.
- Registrar dependencias, decisiones técnicas, limitaciones conocidas y evidencia de validación.

## F01 — System Specification
**Objetivo:** definir propósito, alcance, usuarios, universo inicial de activos, horizontes y outputs esperados de C-MATH-AI.

**Realizar:** documentar la misión, la cadena DATA→QUALITY→QUANT/ML→DECISION→PORTFOLIO→RISK→BACKTEST→PAPER; especificar DecisionRecord, datos de mercado/fundamentales/macro, requisitos de calidad, trazabilidad, reproducibilidad y restricciones de seguridad.

**Entregables:** SYSTEM_SPEC.md, vocabulario de dominio, principios de diseño y límites explícitos del producto.

**Cierre:** requisitos verificables, salidas definidas, prohibiciones de look-ahead y bypass de riesgo documentadas.

## F02 — Software Architecture
**Objetivo:** convertir la especificación en límites de módulos y dependencias mantenibles.

**Realizar:** definir arquitectura modular por capas/ports-and-adapters, responsabilidades de core, data, quant, ml, signals, portfolio, risk, backtest, execution y apps; establecer contratos y dirección de dependencias.

**Entregables:** docs/ARCHITECTURE.md, estructura de paquetes objetivo y reglas de autoridad.

**Cierre:** cada módulo tiene responsabilidad clara; core y quant no dependen de UI, proveedores concretos o brokers; ejecución exige aprobación de riesgo.

## F03 — Repository Initialization
**Objetivo:** preparar un repositorio Python reproducible.

**Realizar:** inicializar paquetes, configuración de build, estructura de tests, tooling de desarrollo, comandos estándar y descubrimiento de paquetes; evitar que los imports dependan del directorio de ejecución.

**Entregables:** pyproject.toml, estructura de directorios, configuración de herramientas y comandos de desarrollo.

**Cierre:** instalación limpia, imports reproducibles y comandos básicos de calidad ejecutables.

## F04 — Domain Contracts
**Objetivo:** definir tipos compartidos y validaciones de dominio antes de integrar módulos.

**Realizar:** crear/exportar contratos para series, retornos, activos, datos, señales, forecasts, métricas y objetos de cartera/backtest según el alcance existente; validar invariantes de fechas, longitudes, unidades y estados.

**Entregables:** core/contracts, exportaciones públicas y documentación de contratos.

**Cierre:** los módulos intercambian datos tipados; estados inválidos se rechazan explícitamente y los contratos tienen tests.

## F05 — Configuration and Environments
**Objetivo:** gestionar configuración de forma explícita y segura.

**Realizar:** separar defaults seguros, configuración por entorno y variables secretas; proporcionar plantilla pública de variables; validar tipos, rangos y valores obligatorios; evitar credenciales en el repositorio.

**Entregables:** configuración central, plantilla de entorno y documentación de ejecución local/CI.

**Cierre:** configuración validada al inicio, secretos fuera del código y comportamiento reproducible por entorno.

## F06 — Quality Gates and CI Foundation
**Objetivo:** establecer una base automática de calidad y seguridad.

**Realizar:** configurar tests, lint, formato, tipado cuando aplique, cobertura informativa, auditoría de dependencias y workflows; documentar cómo reproducir fallos localmente.

**Entregables:** workflows de CI/Security, configuración de herramientas y guías de desarrollo.

**Cierre:** los gates corren sobre los cambios pertinentes, reportan errores reales y no ocultan fallos con skips o exclusiones injustificadas.

## F07 — Market Data Engine
**Objetivo:** establecer la ingestión y el contrato de datos de mercado.

**Realizar:** definir modelos de OHLCV/series de precios, interfaces de proveedor y adaptadores; validar timestamps, duplicados, orden temporal, valores inválidos, frecuencia y metadatos de procedencia.

**Entregables:** capa data de mercado, contratos, tests de calidad y documentación del motor.

**Cierre:** datos reproducibles y normalizados; proveedores intercambiables; series defectuosas se rechazan o marcan explícitamente.

## F08 — Fundamental Data Engine
**Objetivo:** representar estados financieros y datos fundamentales con contexto temporal.

**Realizar:** modelar ingresos, beneficios, balance, deuda, caja, acciones, dividendos y flujos; preservar periodos fiscales, moneda, unidades, fecha de publicación y fuente; distinguir periodo reportado de fecha de disponibilidad.

**Entregables:** contratos/adaptadores fundamentales, validaciones, tests y documentación.

**Cierre:** snapshots fundamentales tipados y point-in-time safe; no se usa un dato antes de que estuviera publicado.

## F09 — Macro Data Engine
**Objetivo:** gestionar series macroeconómicas y de tipos con frecuencia y revisiones explícitas.

**Realizar:** definir series macro, fechas de observación/publicación, unidades, frecuencia, país/región y proveedor; manejar missing data, cambios de frecuencia y revisiones sin introducir información futura.

**Entregables:** capa de datos macro, contratos, validaciones y documentación.

**Cierre:** series macro trazables y alineadas por disponibilidad temporal, con unidades y frecuencia declaradas.

## F10 — Corporate Actions
**Objetivo:** modelar eventos corporativos que alteran precios, unidades o flujos.

**Realizar:** representar splits, reverse splits, dividendos, spin-offs, fusiones y cambios de ticker; definir ajustes y procedencia; prevenir doble ajuste entre proveedor y pipeline.

**Entregables:** modelos/procesamiento de corporate actions, pruebas de casos conocidos y documentación.

**Cierre:** transformaciones reproducibles, sin doble conteo y con trazabilidad del evento aplicado.

## F11 — Canonical Normalization Layer
**Objetivo:** convertir datos de distintas fuentes a una representación común.

**Realizar:** normalizar esquemas, timestamps/timezones, identificadores de activos, monedas, unidades, frecuencias y convenciones; mantener linaje hacia los datos originales y distinguir dato ausente de dato inválido.

**Entregables:** capa de normalización, contratos canónicos, validaciones y tests de integración.

**Cierre:** fuentes equivalentes producen estructuras canónicas consistentes; conversiones no son silenciosas ni ambiguas.

## F12 — Data Quality and Feature Store
**Objetivo:** validar datasets y almacenar features con versión y procedencia.

**Realizar:** comprobar faltantes, duplicados, stale data, valores imposibles, consistencia temporal y leakage; definir estado de calidad; versionar features, parámetros, ventana de cálculo y snapshot de origen.

**Entregables:** data quality checks, feature store versionado, contratos y tests.

**Cierre:** cada feature es reproducible, point-in-time safe y rastreable a su dataset y versión.

## F13 — Numerical Hardening
**Objetivo:** fortalecer las primitivas numéricas frente a bordes y datos patológicos.

**Realizar:** especificar comportamiento ante NaN/inf, denominadores cero, muestras insuficientes, overflow/underflow, precisión y tolerancias; añadir pruebas de propiedades y regresión; documentar convenciones numéricas.

**Entregables:** validaciones numéricas, tests robustos y guía de convenciones.

**Cierre:** entradas inválidas fallan de forma explícita, las tolerancias están justificadas y los casos límite no producen resultados engañosos.

## F14 — Returns and Performance Mathematics
**Objetivo:** proporcionar cálculos consistentes de retornos y rendimiento.

**Realizar:** implementar/normalizar retornos simples y logarítmicos, acumulación, rendimiento compuesto/CAGR, retornos rolling y comparativos según contratos existentes; definir convención de frecuencia, anualización y tratamiento de valores inválidos.

**Entregables:** módulo cuantitativo de retornos/performance, contratos, tests numéricos y documentación.

**Cierre:** fórmulas verificadas contra ejemplos de referencia; frecuencia y anualización explícitas; sin mezcla de retornos simples y logarítmicos.

## F15 — Statistical Engine
**Objetivo:** establecer estadística descriptiva y herramientas estadísticas reutilizables.

**Realizar:** cubrir estadísticas básicas sobre series (media, varianza, desviación, covarianza/correlación, cuantiles, asimetría/curtosis y medidas relacionadas según el alcance implementado); definir ddof, muestras mínimas, NaN y tolerancias; exportar contratos públicos.

**Entregables:** módulo estadístico, contratos, tests numéricos, documentación y workflow de validación específico.

**Cierre:** resultados contrastados con referencias conocidas, casos degenerados definidos y validación de la fase integrada en CI.

## Secuencia posterior

F16–F42 están descritas individualmente en [ROADMAP_F16_F42.md](ROADMAP_F16_F42.md), incluyendo objetivo, alcance, reglas e invariantes y criterio de cierre de cada fase.

## Política de estado

El registro del roadmap no equivale a certificación de implementación ni de CI. Para marcar una fase como cerrada deben existir implementación comprometida, contratos adecuados, tests relevantes, documentación, workflow pertinente verde en el HEAD vigente y ausencia de regresiones conocidas que afecten a fases dependientes.
