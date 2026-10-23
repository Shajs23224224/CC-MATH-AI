# F31 — Deep Learning

## Status and scope

F31 introduces a bounded, optional PyTorch backend for temporal supervised learning and feature
reconstruction. The backend is not imported at package-import time: users of the existing NumPy /
scikit-learn features do not need PyTorch installed. Install the extra with
\`python -m pip install -e ".[deep-learning]"\`.

The allow-list includes:

| Identifier | Architecture | Intended use |
| --- | --- | --- |
| \`mlp\` | Multilayer perceptron over a flattened input window | Tabular and short-window baselines |
| \`lstm\` | Long short-term memory recurrent network | Temporal regression, classification and forecasting |
| \`gru\` | Gated recurrent unit network | A smaller recurrent baseline |
| \`tcn\` | Dilated temporal convolution network | Local and multi-scale temporal patterns |
| \`attention\` | Multi-head self-attention encoder | Window-level context aggregation |
| \`transformer\` | Transformer encoder | Temporal representation learning |
| \`temporal_fusion_transformer\` | Compact TFT-inspired gated recurrent/attention model | Temporal feature selection and fusion |
| \`autoencoder\` | Dense encoder-decoder | Feature-window reconstruction |
| \`vae\` | Variational encoder-decoder with KL regularization | Stochastic latent representation and reconstruction |

The temporal-fusion implementation is explicitly TFT-inspired; it is not the complete canonical
Temporal Fusion Transformer. It does not implement static covariate encoders, quantile heads or
the complete interpretation toolkit. These must not be claimed from the current API.

The current facade reuses the existing `MLDataset` contract. For autoencoder/VAE mode, the target
column is required by that shared contract but is ignored by training; reconstruction loss is
computed from the feature windows themselves.

## API

Use the typed contract and model facade:

\`\`\`python
from core.contracts.deep_learning import DeepLearningConfig
from core.contracts.ml import MLDataset
from ml import DeepLearningModel

dataset = MLDataset(
    dataset_id="synthetic-next-value-v1",
    source="synthetic example",
    feature_names=("trend", "cycle"),
    target_name="next_value",
    features=tuple((float(i), float(i % 7)) for i in range(160)),
    target=tuple(0.02 * i + 0.1 * (i % 7) for i in range(160)),
    task="regression",
)
model = DeepLearningModel(
    DeepLearningConfig(
        algorithm="lstm",
        task="forecasting",
        sequence_length=8,
        random_seed=31,
        epochs=50,
        patience=8,
    )
)
report = model.fit(dataset)
next_values = model.predict(dataset.features[-8:])
\`\`\`

The dataset rows must already be ordered in the permitted temporal sequence. For supervised
models, each input window uses only the preceding \`sequence_length\` rows and predicts the target
at the next row. The supplied target must therefore represent the value that is actually intended
to be predicted at that alignment.

## Temporal evaluation and leakage controls

For a dataset of \(N\) rows, F31 creates consecutive train, validation and final test segments.
The default fractions are 65% training, 15% validation and 20% test. The validation segment is
used only for early stopping/checkpoint selection. The final test segment is evaluated once after
the best validation checkpoint is restored; it is not used for model selection.

- The \`StandardScaler\` is fitted only on feature rows available before the first validation target.
- Sequence windows are built so each supervised input ends before its target row.
- Classification labels are mapped using training-prefix classes only. A validation/test label
  absent from training is rejected rather than silently extending the model's vocabulary.
- Gradients are clipped, seeds are recorded, and parameter, epoch, observation, feature and window
  memory budgets are enforced.
- Checkpoints are held in memory only. No executable model objects or weights are put in the
  result contract; persistence and deployment need separate reviewed contracts.

The result records split boundaries, scaler scope/statistics, seed, PyTorch version, parameter
count, training fingerprint, best epoch, validation loss and out-of-sample metrics. Regression
reports MAE/RMSE/R²; classification reports accuracy and macro-F1; reconstruction reports test
reconstruction MSE. Non-finite values and resource-budget violations are rejected.

## Limits and acceptance criteria

This is an extensible model engine, not evidence that deep networks outperform classical baselines.
Synthetic tests verify implementation behavior, deterministic configuration, split discipline,
shapes, inference and resource bounds; they do not establish market alpha or economic validity.
No model is automatically promoted to an ensemble and no trading order can be emitted by this
module.

Before F31 can be marked integrated, its dedicated workflow, general Python 3.11/3.12 CI,
Security checks and regression tests must pass on the final PR head. Baselines, temporal
out-of-sample validation and documented limitations are mandatory; test metrics cannot be used to
choose the architecture or training configuration.
