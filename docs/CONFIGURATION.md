# F05 — Configuration, Secrets and Environments

## Objective

Provide explicit environment separation and a safe configuration boundary.

## Environments

C-MATH-AI defines four logical environments:

| Environment | Intended use | Live execution |
|---|---|---|
| development | Local development and experimentation | prohibited |
| testing | Automated tests and validation | prohibited |
| paper | Simulated trading | prohibited |
| production | Controlled deployment | disabled by default |

Live execution requires all of the following:

1. `APP_ENV=production`;
2. `LIVE_EXECUTION_ENABLED=true`;
3. an explicit broker provider;
4. broker credentials injected outside source control.

## Secret policy

Never commit:

- market-data API keys;
- broker API keys;
- broker API secrets;
- database credentials;
- access tokens;
- private keys.

Secrets are loaded from environment variables through `CMathSettings.from_environment()` and represented internally with Pydantic `SecretStr` where applicable.

The repository contains templates only.

## Supported environment variables

- `APP_ENV`
- `LOG_LEVEL`
- `MARKET_DATA_PROVIDER`
- `MARKET_DATA_API_KEY`
- `DATABASE_URL`
- `BROKER_PROVIDER`
- `BROKER_API_KEY`
- `BROKER_API_SECRET`
- `LIVE_EXECUTION_ENABLED`
- `RISK_MAX_POSITION_WEIGHT`
- `RISK_MAX_PORTFOLIO_LEVERAGE`
- `RISK_MAX_DAILY_LOSS`

## Precedence

Runtime secrets and deployment-specific overrides come from environment variables.

TOML environment profiles provide public defaults and documentation. They must never become a secret store.

## Safety rules

- Live execution is disabled by default.
- Paper/testing/development environments cannot enable live execution.
- Production live execution requires a broker provider and both broker credentials.
- Secret values must not be logged.
- Configuration objects are immutable after validation.
- Invalid environment values fail fast.

## Files

- `core/config/settings.py` — typed runtime settings and safety validation.
- `core/config/loader.py` — non-secret TOML loading and environment settings loader.
- `config/environments/*.toml` — public environment profiles.
- `.env.example` — variable template without credentials.
