# F20 — Volatility & Stochastic

F20 adds the deterministic volatility and stochastic-process layer specified in the roadmap. It remains upstream of the future Risk Engine (F21): these models estimate uncertainty and dynamics but do not authorize, reduce or block any position.

## Scope

Implemented primitives:

- rolling historical volatility;
- EWMA volatility;
- ARCH(1);
- GARCH(1,1);
- EGARCH(1,1);
- GJR-GARCH(1,1);
- a deterministic stochastic-volatility log-variance AR(1) quasi-likelihood model;
- DCC-GARCH(1,1) with GARCH margins;
- Ornstein-Uhlenbeck estimation;
- generic mean-reversion estimation as the OU core.

Annualization is explicit and is applied only as a square-root-of-frequency scaling of conditional variance.

## Numerical methodology

### Historical and EWMA volatility

rolling_volatility computes the population standard deviation over each trailing window. ewma_volatility updates variance recursively:

sigma2_t = lambda * sigma2_(t-1) + (1-lambda) * r_t^2

with lambda in (0,1).

### ARCH / GARCH

ARCH(1) and GARCH(1,1) are fitted by constrained Gaussian quasi-maximum likelihood. The parameter restrictions enforce positive variance and the covariance-stationarity condition for GARCH:

alpha + beta < 1.

### EGARCH

EGARCH models log variance and permits asymmetric response to standardized shocks. The implementation constrains abs(beta) < 1 and preserves the sign-sensitive term.

### GJR-GARCH

GJR-GARCH adds a non-negative response to negative shocks. The implementation enforces:

alpha + 0.5 gamma + beta < 1.

### Stochastic volatility

The stochastic-volatility primitive estimates an AR(1) process for log squared returns. It is intentionally documented as a Gaussian quasi-likelihood state proxy rather than a full latent-state particle filter or MCMC estimator.

### DCC-GARCH

Each series receives a GARCH(1,1) conditional variance. Standardized residuals feed a DCC recursion:

Q_t = (1-a-b) Q_bar + a z_(t-1) z'_(t-1) + b Q_(t-1)

with a,b >= 0 and a+b < 1. The public contract exposes the one-step correlation matrix.

### Ornstein-Uhlenbeck / mean reversion

OU parameters are estimated from the discrete regression:

x_t = c + phi x_(t-1) + epsilon_t

with 0 < phi < 1, then mapped to continuous time:

kappa = -ln(phi) / Delta_t

theta = c / (1-phi)

and:

half-life = ln(2) / kappa.

## Contracts

- VolatilityFitSummary: fit statistics and convergence.
- VolatilityForecast: finite non-negative annualized volatility forecasts.
- DynamicCorrelationForecast: symmetric correlation matrix.
- MeanReversionFit: speed, long-run mean, diffusion and half-life.
- MeanReversionForecast: deterministic conditional mean path.

## Validation rules

- finite inputs only;
- explicit sample minima;
- positive annualization factor;
- positive forecast horizon;
- constrained volatility parameters;
- covariance-stationarity restrictions;
- no silent imputation;
- no look-ahead;
- no BUY/HOLD/SELL output;
- no Risk Governor bypass.

F20 is the model/evidence layer. Portfolio and risk decisions remain subsequent phases.
