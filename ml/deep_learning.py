"""Bounded, reproducible deep-learning models with chronological evaluation (F31)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from core.contracts.deep_learning import DeepLearningConfig, DeepLearningResult
from core.contracts.ml import MLDataset, MLPrediction

from .errors import MLInputError, MLPredictionError, MLTrainingError

_RECONSTRUCTION_ALGORITHMS = {"autoencoder", "vae"}
_FLATTENED_ALGORITHMS = {"mlp", "autoencoder", "vae"}
_MAX_WINDOW_VALUES = 2_000_000


def _load_torch() -> Any:
    """Load PyTorch only when a caller explicitly uses the optional F31 engine."""
    try:
        return import_module("torch")
    except ImportError as exc:
        raise MLInputError(
            "F31 requires the optional PyTorch backend; install c-math-ai[deep-learning]"
        ) from exc


def supported_deep_learning_algorithms() -> dict[str, tuple[str, ...]]:
    """Return the explicit registry of supported architecture families."""
    return {
        "supervised": (
            "attention",
            "gru",
            "lstm",
            "mlp",
            "temporal_fusion_transformer",
            "tcn",
            "transformer",
        ),
        "reconstruction": ("autoencoder", "vae"),
    }


def _build_network(
    torch: Any,
    config: DeepLearningConfig,
    feature_count: int,
    output_size: int,
) -> Any:
    """Construct one architecture from the closed allow-list."""
    nn = torch.nn
    sequence_length = config.sequence_length
    hidden = config.hidden_dim
    input_size = sequence_length * feature_count

    if config.algorithm == "mlp":
        layers: list[Any] = []
        width = input_size
        for _ in range(config.num_layers):
            layers.extend([nn.Linear(width, hidden), nn.GELU()])
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            width = hidden
        layers.append(nn.Linear(width, output_size))
        return nn.Sequential(*layers)

    if config.algorithm in _RECONSTRUCTION_ALGORITHMS:
        latent_size = max(2, hidden // 2)

        if config.algorithm == "autoencoder":

            class DenseAutoencoder(nn.Module):  # type: ignore[name-defined]
                def __init__(self) -> None:
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_size, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, latent_size),
                        nn.GELU(),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_size, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, input_size),
                    )

                def forward(self, values: Any) -> Any:
                    return self.decoder(self.encoder(values))

            return DenseAutoencoder()

        class VariationalAutoencoder(nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.encoder_body = nn.Sequential(
                    nn.Linear(input_size, hidden),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                self.mean = nn.Linear(hidden, latent_size)
                self.log_variance = nn.Linear(hidden, latent_size)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_size, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, input_size),
                )

            def forward(self, values: Any) -> tuple[Any, Any, Any]:
                encoded = self.encoder_body(values)
                mean = self.mean(encoded)
                log_variance = torch.clamp(self.log_variance(encoded), min=-12.0, max=8.0)
                if self.training:
                    deviation = torch.exp(0.5 * log_variance)
                    latent = mean + torch.randn_like(deviation) * deviation
                else:
                    latent = mean
                return self.decoder(latent), mean, log_variance

        return VariationalAutoencoder()

    if config.algorithm in {"lstm", "gru"}:
        recurrent_type = nn.LSTM if config.algorithm == "lstm" else nn.GRU

        class RecurrentNetwork(nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.recurrent = recurrent_type(
                    input_size=feature_count,
                    hidden_size=hidden,
                    num_layers=config.num_layers,
                    dropout=config.dropout if config.num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden, output_size)
                self.dropout = nn.Dropout(config.dropout)

            def forward(self, values: Any) -> Any:
                encoded, _ = self.recurrent(values)
                return self.head(self.dropout(encoded[:, -1, :]))

        return RecurrentNetwork()

    if config.algorithm == "tcn":

        class TemporalConvNetwork(nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                blocks: list[Any] = []
                in_channels = feature_count
                for layer_index in range(config.num_layers):
                    dilation = 2**layer_index
                    blocks.extend(
                        [
                            nn.Conv1d(
                                in_channels,
                                hidden,
                                kernel_size=3,
                                padding=dilation,
                                dilation=dilation,
                            ),
                            nn.GELU(),
                            nn.Dropout(config.dropout),
                        ]
                    )
                    in_channels = hidden
                self.temporal = nn.Sequential(*blocks)
                self.head = nn.Linear(hidden, output_size)

            def forward(self, values: Any) -> Any:
                encoded = self.temporal(values.transpose(1, 2))
                return self.head(encoded[:, :, -1])

        return TemporalConvNetwork()

    if config.algorithm == "attention":

        class AttentionNetwork(nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Linear(feature_count, hidden)
                self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden))
                self.attention = nn.MultiheadAttention(
                    hidden, config.num_heads, dropout=config.dropout, batch_first=True
                )
                self.norm = nn.LayerNorm(hidden)
                self.head = nn.Linear(hidden, output_size)

            def forward(self, values: Any) -> Any:
                embedded = self.projection(values) + self.position
                attended, _ = self.attention(embedded, embedded, embedded, need_weights=False)
                return self.head(self.norm(embedded + attended)[:, -1, :])

        return AttentionNetwork()

    if config.algorithm == "transformer":

        class TransformerNetwork(nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Linear(feature_count, hidden)
                self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden))
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=config.num_heads,
                    dim_feedforward=hidden * 4,
                    dropout=config.dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
                self.head = nn.Linear(hidden, output_size)

            def forward(self, values: Any) -> Any:
                embedded = self.projection(values) + self.position
                return self.head(self.encoder(embedded)[:, -1, :])

        return TransformerNetwork()

    if config.algorithm == "temporal_fusion_transformer":

        class TemporalFusionNetwork(nn.Module):  # type: ignore[name-defined]
            """Compact TFT-inspired network; intentionally not the full canonical TFT."""

            def __init__(self) -> None:
                super().__init__()
                self.variable_selection = nn.Linear(feature_count, feature_count)
                self.projection = nn.Linear(feature_count, hidden)
                self.recurrent = nn.LSTM(
                    hidden,
                    hidden,
                    num_layers=config.num_layers,
                    dropout=config.dropout if config.num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.attention = nn.MultiheadAttention(
                    hidden, config.num_heads, dropout=config.dropout, batch_first=True
                )
                self.gate = nn.Linear(hidden * 2, hidden)
                self.norm = nn.LayerNorm(hidden)
                self.head = nn.Linear(hidden, output_size)

            def forward(self, values: Any) -> Any:
                weights = torch.softmax(self.variable_selection(values), dim=-1)
                selected = self.projection(values * weights)
                temporal, _ = self.recurrent(selected)
                attended, _ = self.attention(temporal, temporal, temporal, need_weights=False)
                latest_temporal = temporal[:, -1, :]
                latest_attention = attended[:, -1, :]
                gate = torch.sigmoid(
                    self.gate(torch.cat([latest_temporal, latest_attention], dim=-1))
                )
                fused = gate * latest_temporal + (1.0 - gate) * latest_attention
                return self.head(self.norm(fused))

        return TemporalFusionNetwork()

    raise MLInputError(f"unsupported deep-learning algorithm: {config.algorithm}")


def _as_matrix(
    rows: Sequence[Sequence[float]],
    *,
    feature_count: int | None = None,
    max_observations: int,
) -> np.ndarray:
    """Validate a bounded finite feature matrix before scaling or model execution."""
    try:
        matrix = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MLInputError("features must be a rectangular numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise MLInputError("features must be a non-empty two-dimensional matrix")
    if feature_count is not None and matrix.shape[1] != feature_count:
        raise MLInputError(f"expected {feature_count} features per row")
    if matrix.shape[0] > max_observations:
        raise MLInputError(f"observations exceed the configured maximum of {max_observations}")
    if matrix.shape[1] > 64:
        raise MLInputError("at most 64 features are supported")
    if not np.all(np.isfinite(matrix)):
        raise MLInputError("features must contain only finite values")
    return matrix


def _windows(matrix: np.ndarray, sequence_length: int) -> np.ndarray:
    """Build windows where each input contains only observations before its target index."""
    if len(matrix) <= sequence_length:
        raise MLInputError("not enough rows to construct the configured sequence windows")
    return np.asarray(
        [matrix[index - sequence_length : index] for index in range(sequence_length, len(matrix))],
        dtype=np.float32,
    )


class DeepLearningModel:
    """Train one bounded neural model using a chronological train/validation/test protocol."""

    def __init__(self, config: DeepLearningConfig) -> None:
        self.config = config
        self.result: DeepLearningResult | None = None
        self._network: Any | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._classes: tuple[float | str, ...] | None = None
        self._torch: Any | None = None
        self._device: Any | None = None

    def _forward_prediction(self, network: Any, features: Any) -> Any:
        output = network(features)
        if self.config.algorithm == "vae":
            return output[0]
        return output

    def _loss(self, torch: Any, prediction: Any, target: Any) -> Any:
        if self.config.algorithm == "vae":
            reconstruction, mean, log_variance = prediction
            reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, target)
            kl_loss = -0.5 * torch.mean(
                torch.sum(1.0 + log_variance - mean.square() - log_variance.exp(), dim=1)
            )
            return reconstruction_loss + 0.001 * torch.clamp(kl_loss, min=0.0)
        if self.config.task == "classification":
            return torch.nn.functional.cross_entropy(prediction, target)
        return torch.nn.functional.mse_loss(prediction, target)

    def fit(self, dataset: MLDataset) -> DeepLearningResult:
        """Fit only on the chronological training segment; use validation for early stopping."""
        torch = _load_torch()
        matrix = _as_matrix(
            dataset.features,
            feature_count=len(dataset.feature_names),
            max_observations=self.config.max_observations,
        )
        if self.config.task == "classification" and dataset.task != "classification":
            raise MLInputError("classification models require a classification MLDataset")
        if self.config.task in {"regression", "forecasting"} and dataset.task != "regression":
            raise MLInputError("regression and forecasting models require a regression MLDataset")
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise MLInputError("CUDA was requested but is unavailable in this PyTorch runtime")

        sequence_length = self.config.sequence_length
        if len(matrix) * sequence_length * matrix.shape[1] > _MAX_WINDOW_VALUES:
            raise MLInputError(
                "sequence windows exceed the bounded memory budget; reduce sequence_length, "
                "feature count or observation count"
            )
        train_end_fraction = 1.0 - self.config.test_fraction - self.config.validation_fraction
        train_end = int(math.floor(len(matrix) * train_end_fraction))
        validation_end = int(math.floor(len(matrix) * (1.0 - self.config.test_fraction)))
        if train_end - sequence_length < 8:
            raise MLInputError(
                "training segment must contain at least eight sequence-target samples"
            )
        if validation_end - train_end < 2 or len(matrix) - validation_end < 2:
            raise MLInputError("validation and test segments must each contain at least two rows")

        # Fit preprocessing only to features available before the first validation target.
        scaler_fit_end = train_end - 1
        if scaler_fit_end < sequence_length + 1:
            raise MLInputError("training prefix is too short to fit the feature scaler")
        scaler = StandardScaler()
        try:
            scaler.fit(matrix[:scaler_fit_end])
            scaled = scaler.transform(matrix).astype(np.float32)
        except (ValueError, TypeError, FloatingPointError) as exc:
            raise MLTrainingError("training-prefix feature scaling failed") from exc

        sequence_windows = _windows(scaled, sequence_length)
        target_indices = np.arange(sequence_length, len(matrix), dtype=np.int64)
        train_mask = target_indices < train_end
        validation_mask = (target_indices >= train_end) & (target_indices < validation_end)
        test_mask = target_indices >= validation_end
        train_x = sequence_windows[train_mask]
        validation_x = sequence_windows[validation_mask]
        test_x = sequence_windows[test_mask]
        if min(len(train_x), len(validation_x), len(test_x)) < 2:
            raise MLInputError("each chronological segment must contain at least two windows")

        target_values = list(dataset.target)
        classes: tuple[float | str, ...] | None = None
        if self.config.task == "reconstruction":
            train_y_array = train_x.reshape(len(train_x), -1)
            validation_y_array = validation_x.reshape(len(validation_x), -1)
            test_y_array = test_x.reshape(len(test_x), -1)
            output_size = sequence_length * matrix.shape[1]
        elif self.config.task == "classification":
            train_labels = target_values[sequence_length:train_end]
            classes = tuple(
                sorted(set(train_labels), key=lambda value: (type(value).__name__, str(value)))
            )
            if len(classes) < 2:
                raise MLInputError("training prefix must contain at least two target classes")
            class_to_index = {label: index for index, label in enumerate(classes)}
            try:
                encoded = np.asarray(
                    [class_to_index[target_values[index]] for index in target_indices],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise MLInputError(
                    "validation/test contains a class absent from the training prefix"
                ) from exc
            train_y_array = encoded[train_mask]
            validation_y_array = encoded[validation_mask]
            test_y_array = encoded[test_mask]
            output_size = len(classes)
        else:
            numeric_targets = np.asarray(target_values, dtype=np.float32)
            if not np.all(np.isfinite(numeric_targets)):
                raise MLInputError("regression targets must be finite")
            train_y_array = numeric_targets[target_indices[train_mask]].reshape(-1, 1)
            validation_y_array = numeric_targets[target_indices[validation_mask]].reshape(-1, 1)
            test_y_array = numeric_targets[target_indices[test_mask]].reshape(-1, 1)
            output_size = 1

        torch.manual_seed(self.config.random_seed)
        if self.config.device == "cuda":
            torch.cuda.manual_seed_all(self.config.random_seed)
        device = torch.device(self.config.device)
        network = _build_network(torch, self.config, matrix.shape[1], output_size).to(device)
        parameter_count = sum(int(parameter.numel()) for parameter in network.parameters())
        if parameter_count > self.config.max_parameters:
            raise MLInputError(
                f"network has {parameter_count} parameters; configured maximum is "
                f"{self.config.max_parameters}"
            )

        def as_input(values: np.ndarray) -> Any:
            tensor = torch.tensor(values, dtype=torch.float32, device=device)
            if self.config.algorithm in _FLATTENED_ALGORITHMS:
                return tensor.reshape(tensor.shape[0], -1)
            return tensor

        train_input = as_input(train_x)
        validation_input = as_input(validation_x)
        test_input = as_input(test_x)
        target_dtype = torch.long if self.config.task == "classification" else torch.float32
        train_target = torch.tensor(train_y_array, dtype=target_dtype, device=device)
        validation_target = torch.tensor(validation_y_array, dtype=target_dtype, device=device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.random_seed)
        train_dataset = torch.utils.data.TensorDataset(train_input, train_target)
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(self.config.batch_size, len(train_input)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best_loss = math.inf
        best_epoch = 0
        epochs_completed = 0
        stale_epochs = 0
        best_state: Any | None = None

        for epoch in range(1, self.config.epochs + 1):
            network.train()
            for batch_features, batch_target in loader:
                optimizer.zero_grad(set_to_none=True)
                raw_output = network(batch_features)
                loss = self._loss(torch, raw_output, batch_target)
                loss_value = float(loss.detach().cpu().item())
                if not math.isfinite(loss_value):
                    raise MLTrainingError("deep-learning training loss became non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()

            network.eval()
            with torch.no_grad():
                validation_output = network(validation_input)
                validation_loss = float(
                    self._loss(torch, validation_output, validation_target).detach().cpu().item()
                )
            if not math.isfinite(validation_loss):
                raise MLTrainingError("deep-learning validation loss became non-finite")
            epochs_completed = epoch
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(network.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.config.patience:
                break

        if best_state is None:
            raise MLTrainingError("training did not produce a valid checkpoint")
        network.load_state_dict(best_state)
        network.eval()
        with torch.no_grad():
            test_output = self._forward_prediction(network, test_input)
            test_array = test_output.detach().cpu().numpy()
            if not np.all(np.isfinite(test_array)):
                raise MLTrainingError("deep-learning model produced non-finite test output")

        metrics: dict[str, float]
        test_predictions: tuple[float | str, ...] = ()
        test_targets: tuple[float | str, ...] = ()
        if self.config.task == "reconstruction":
            expected = test_y_array
            reconstruction_mse = float(
                mean_squared_error(expected.reshape(-1), test_array.reshape(-1))
            )
            metrics = {"reconstruction_mse": reconstruction_mse}
        elif self.config.task == "classification":
            assert classes is not None
            predicted_indices = np.argmax(test_array, axis=1)
            target_indices_array = np.asarray(test_y_array, dtype=np.int64)
            metrics = {
                "accuracy": float(accuracy_score(target_indices_array, predicted_indices)),
                "f1_macro": float(
                    f1_score(
                        target_indices_array,
                        predicted_indices,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
            test_predictions = tuple(classes[int(index)] for index in predicted_indices)
            test_targets = tuple(
                target_values[int(index)] for index in target_indices[test_mask].tolist()
            )
        else:
            prediction_values = test_array.reshape(-1)
            target_values_array = np.asarray(test_y_array, dtype=np.float64).reshape(-1)
            metrics = {
                "mae": float(mean_absolute_error(target_values_array, prediction_values)),
                "rmse": float(
                    math.sqrt(mean_squared_error(target_values_array, prediction_values))
                ),
                "r2": float(r2_score(target_values_array, prediction_values)),
            }
            test_predictions = tuple(float(value) for value in prediction_values.tolist())
            test_targets = tuple(float(value) for value in target_values_array.tolist())

        training_prefix = {
            "features": matrix[:scaler_fit_end].tolist(),
            "target": target_values[sequence_length:train_end]
            if self.config.task != "reconstruction"
            else [],
            "feature_names": list(dataset.feature_names),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                training_prefix,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        config_json = self.config.model_dump(mode="json")
        identity = json.dumps(
            {
                "algorithm": self.config.algorithm,
                "seed": self.config.random_seed,
                "dataset_id": dataset.dataset_id,
                "data_fingerprint": fingerprint,
                "config": config_json,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        model_id = f"{self.config.algorithm}:{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
        limitations = [
            "Training, scaler fitting and checkpoint selection use only the chronological training "
            "and validation segments; the final test segment is never used for selection.",
            "Rows must be supplied in chronological order and target timing must match the "
            "caller-defined forecasting horizon; sequence inputs end before the target row.",
            "Passing synthetic tests is not evidence of market alpha or live trading readiness.",
            "Neural weights are held in memory only; checkpoint persistence, deployment and "
            "ensemble promotion require separate reviewed interfaces.",
        ]
        if self.config.algorithm == "temporal_fusion_transformer":
            limitations.append(
                "This is a compact TFT-inspired architecture, not the full canonical TFT with "
                "static covariates, quantile heads and a production-grade interpretation toolkit."
            )
        if self.config.task == "reconstruction":
            limitations.append(
                "Reconstruction scores measure feature reconstruction error; they are not labels, "
                "causal explanations or direct trading signals."
            )
        result = DeepLearningResult(
            model_id=model_id,
            dataset_id=dataset.dataset_id,
            algorithm=self.config.algorithm,
            task=self.config.task,
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            total_observations=len(matrix),
            training_samples=len(train_x),
            validation_samples=len(validation_x),
            test_samples=len(test_x),
            train_end_index=train_end,
            validation_start_index=train_end,
            validation_end_index=validation_end,
            test_start_index=validation_end,
            test_end_index=len(matrix),
            sequence_length=sequence_length,
            scaler_fit_end_index=scaler_fit_end,
            scaler_mean=tuple(float(value) for value in scaler.mean_.tolist()),
            scaler_scale=tuple(float(value) for value in scaler.scale_.tolist()),
            random_seed=self.config.random_seed,
            device=self.config.device,
            library_version=str(torch.__version__),
            data_fingerprint=fingerprint,
            parameter_count=parameter_count,
            epochs_requested=self.config.epochs,
            epochs_completed=epochs_completed,
            best_epoch=best_epoch,
            early_stopped=epochs_completed < self.config.epochs,
            best_validation_loss=best_loss,
            metrics=metrics,
            test_predictions=test_predictions,
            test_targets=test_targets,
            limitations=tuple(limitations),
        )
        self._network = network
        self._scaler = scaler
        self._feature_names = dataset.feature_names
        self._classes = classes
        self._torch = torch
        self._device = device
        self.result = result
        return result

    def predict(self, features: Sequence[Sequence[float]]) -> MLPrediction:
        """Predict the next target after each supplied feature window."""
        if (
            self._network is None
            or self._scaler is None
            or self._torch is None
            or self.result is None
            or self._feature_names is None
        ):
            raise MLPredictionError("fit must be called before predict")
        if self.config.task == "reconstruction":
            raise MLPredictionError("use reconstruct() for autoencoder and VAE models")
        matrix = _as_matrix(
            features,
            feature_count=len(self._feature_names),
            max_observations=self.config.max_observations,
        )
        if len(matrix) < self.config.sequence_length:
            raise MLInputError("prediction input must contain at least sequence_length rows")
        if len(matrix) * self.config.sequence_length * matrix.shape[1] > _MAX_WINDOW_VALUES:
            raise MLInputError(
                "sequence windows exceed the bounded memory budget; reduce sequence_length, "
                "feature count or observation count"
            )
        scaled = self._scaler.transform(matrix).astype(np.float32)
        windows = np.asarray(
            [
                scaled[index - self.config.sequence_length : index]
                for index in range(self.config.sequence_length, len(scaled) + 1)
            ],
            dtype=np.float32,
        )
        torch = self._torch
        tensor = torch.tensor(windows, dtype=torch.float32, device=self._device)
        if self.config.algorithm in _FLATTENED_ALGORITHMS:
            tensor = tensor.reshape(tensor.shape[0], -1)
        self._network.eval()
        with torch.no_grad():
            raw = self._forward_prediction(self._network, tensor)
            values = raw.detach().cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise MLPredictionError("deep-learning model produced non-finite predictions")
        if self.config.task == "classification":
            if self._classes is None:
                raise MLPredictionError("classification labels are unavailable")
            stable = values - values.max(axis=1, keepdims=True)
            exponentials = np.exp(stable)
            probabilities_array = exponentials / exponentials.sum(axis=1, keepdims=True)
            labels = np.argmax(probabilities_array, axis=1)
            return MLPrediction(
                predictions=tuple(self._classes[int(index)] for index in labels),
                classes=self._classes,
                probabilities=tuple(
                    tuple(float(value) for value in row) for row in probabilities_array.tolist()
                ),
            )
        predictions = tuple(float(value) for value in values.reshape(-1).tolist())
        return MLPrediction(predictions=predictions)

    def reconstruct(self, features: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        """Return flattened reconstructed windows after fitting an autoencoder or VAE."""
        if (
            self._network is None
            or self._scaler is None
            or self._torch is None
            or self._feature_names is None
        ):
            raise MLPredictionError("fit must be called before reconstruct")
        if self.config.task != "reconstruction":
            raise MLPredictionError("reconstruct() requires an autoencoder or VAE model")
        matrix = _as_matrix(
            features,
            feature_count=len(self._feature_names),
            max_observations=self.config.max_observations,
        )
        if len(matrix) < self.config.sequence_length:
            raise MLInputError("reconstruction input must contain at least sequence_length rows")
        if len(matrix) * self.config.sequence_length * matrix.shape[1] > _MAX_WINDOW_VALUES:
            raise MLInputError(
                "sequence windows exceed the bounded memory budget; reduce sequence_length, "
                "feature count or observation count"
            )
        scaled = self._scaler.transform(matrix).astype(np.float32)
        windows = np.asarray(
            [
                scaled[index - self.config.sequence_length : index]
                for index in range(self.config.sequence_length, len(scaled) + 1)
            ],
            dtype=np.float32,
        )
        tensor = self._torch.tensor(windows, dtype=self._torch.float32, device=self._device)
        tensor = tensor.reshape(tensor.shape[0], -1)
        self._network.eval()
        with self._torch.no_grad():
            output = self._forward_prediction(self._network, tensor)
            array = output.detach().cpu().numpy()
        if not np.all(np.isfinite(array)):
            raise MLPredictionError("reconstruction output contains non-finite values")
        return tuple(tuple(float(value) for value in row) for row in array.tolist())
