# Repository Validation

These records concern code, installation, and data integrity. They are not
EyePACS training results and must not be quoted as model performance.

- `tests.xml`: CPU unit tests, including a cached-prediction workflow using synthetic data.
- `data_content_audit.json`: source CSV agreement and exact-byte duplicate audit.
- `dvc_restore.json`: six representative real images restored with `dvc pull`
  from a new empty cache, with SHA-256 comparisons.

The full image tree is registered in DVC. Near-duplicate image analysis and
512px GPU memory/training throughput measurements have not been performed.
No new research experiment was started during this repository reconstruction.
