# Experimental Results

Use this folder as the central record of experiment outputs.

## Tracks

```text
experimental_results/
├── ours_proposed/
├── official_setting/
└── unified_setting/
```

## What To Store

Each track has the same layout:

```text
logs/          # training logs and run summaries
metrics/       # JSON/CSV metric outputs
figures/       # confusion matrices, plots, and analysis figures
predictions/   # validation/test prediction CSVs
```

## Rule

Every experiment should leave enough information here to reproduce the reported
number: method name, seed, split, selected checkpoint rule, metrics, and command
or script used to produce the result.

Large checkpoints and raw image data should not be committed directly to GitHub.
Track them with DVC or store the server path in the corresponding experiment log.
