from enum import StrEnum


class AssetType(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    FX = "fx"
    RATE = "rate"
    COMMODITY = "commodity"
    CRYPTO = "crypto"


class Frequency(StrEnum):
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1D"
    WEEKLY = "1W"


class SignalType(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class RiskAction(StrEnum):
    ALLOW = "ALLOW"
    REDUCE = "REDUCE"
    BLOCK = "BLOCK"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class DataQualityStatus(StrEnum):
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"
