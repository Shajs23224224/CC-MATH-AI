"""Explicit errors for ML inputs, fitting, prediction and experiment persistence."""


class MLInputError(ValueError):
    """Raised when a dataset or prediction input violates a contract."""


class MLTrainingError(RuntimeError):
    """Raised when an allow-listed estimator cannot be trained."""


class MLPredictionError(RuntimeError):
    """Raised when a fitted model cannot produce a valid prediction."""


class ExperimentPersistenceError(RuntimeError):
    """Raised when experiment records cannot be written or recovered."""
