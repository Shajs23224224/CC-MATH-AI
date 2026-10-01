"""C-MATH-AI error taxonomy."""


class CMathError(Exception):
    """Base class for project-specific errors."""


class DataProviderError(CMathError):
    """Raised when a market-data provider cannot satisfy a request."""


class DataQualityError(CMathError):
    """Raised when required market-data quality invariants fail."""


class ConfigurationError(CMathError):
    """Raised for invalid runtime configuration."""
