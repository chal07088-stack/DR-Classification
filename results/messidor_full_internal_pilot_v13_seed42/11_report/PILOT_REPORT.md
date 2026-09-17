# Messidor-2 Full Internal Pilot v13

Seed: 42. Stage A uses the frozen selected ConvNeXt-S eye model. Correction variants use the exact complete bilateral cohort and existing patient split.

|variant|qwk|mae|harm|\n|---|---:|---:|---:|\n|V0|0.7998|0.2846|0.0000|\n|V1|0.8048|0.2962|0.0538|\n|V2|0.8090|0.2769|0.0269|\n|V3|0.7984|0.2885|0.0115|\n|V4|0.8075|0.2885|0.0269|\n|V5|0.7698|0.3115|0.0577|

## Protocol notes
- Laterality metadata is not sufficiently reliable for image canonicalization; no canonicalization was applied.
- Cached 512 preprocessing was reused; correction training applies no random horizontal flip.
- V4 uses bounded boundary-wise correction with counterfactual endpoint supervision; V5 is an unbounded free residual comparator.