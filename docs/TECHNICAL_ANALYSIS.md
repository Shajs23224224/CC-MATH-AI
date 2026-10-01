# F16 — Technical Analysis

F16 implementa indicadores técnicos deterministas sobre series OHLCV. Las salidas están alineadas con la longitud de la entrada y usan None durante el warm-up.

## Indicadores implementados

SMA, EMA, WMA, Momentum, ROC, RSI, MACD, ATR, Bollinger Bands, Stochastic Oscillator, Williams %R, DMI/ADX, Donchian Channels, VWAP acumulado y cambio de volumen.

## Convenciones

- SMA usa la media aritmética de la ventana.
- EMA se inicializa con SMA y después usa alpha = 2 / (n + 1).
- WMA usa pesos 1..n, con mayor peso al dato reciente.
- RSI y ATR usan suavizado de Wilder.
- Bollinger usa desviación estándar poblacional.
- Stochastic con rango plano devuelve 50; Williams %R devuelve -50.
- VWAP usa precio típico (high + low + close) / 3.
- Los valores sin historia suficiente son None.

## No-lookahead

Cada cálculo usa sólo observaciones hasta el índice actual. Las salidas conservan la posición temporal de la entrada; no se consultan datos posteriores.

## Validación y límites

Se validan finitud, ventanas, coherencia OHLC y volumen no negativo. Los tests cubren referencias numéricas, dominios inválidos, alineación, warm-up y no-lookahead.

F16 no genera señales BUY/HOLD/SELL ni optimiza parámetros. Es una capa matemática determinista para las fases posteriores.
