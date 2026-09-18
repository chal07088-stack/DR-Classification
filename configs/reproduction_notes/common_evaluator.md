# Common evaluator lock

All scalar-regression methods retain their paper-native decision rule: round the raw scalar and clip to grades 0–4. Because Huang, DRGCNN and D-ODR do not emit five-class probabilities, the required `prob_0`…`prob_4` schema is populated with a deterministic distance-softmax over the raw scalar. Those auxiliary values are not used for checkpoint selection, `y_pred`, QWK, MAE, Accuracy or Macro-F1. AOR-DR probabilities are the empirical frequencies of its five official stochastic ordinal samples.

Every Test patient contributes exactly two directional targets. All main metrics use a single shared implementation. Paired change analyses join strictly by `sample_id`; bootstrap resamples patients, preserving the two correlated eyes.
