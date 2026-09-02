# Comparison Protocol

## Dataset

- Dataset: APTOS2019 original 3,662 images
- Split: stratified 60:20:20
- Data split seed: 42

## Training Seeds

- 42
- 123
- 128

## Evaluation Metrics

- Quadratic weighted kappa
- Macro-F1
- Grade 3 F1
- Mean absolute error
- Accuracy

## Official-Setting Comparison

Use the same dataset split, training seed set, validation-based model selection,
and evaluation metrics. Keep each paper's official hyperparameters whenever possible.

## Unified-Setting Comparison

Use the same dataset split, training seed set, validation-based model selection,
evaluation metrics, and the project's common training protocol whenever compatible.
