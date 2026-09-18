# DRGCNN reproduction lock

Source lock: paper *A novel approach for intelligent diagnosis and grading of diabetic retinopathy* and official repository commit `bf80220a5786bf52678519f381b742827447402f`.

The main encoder is ImageNet CAM-EfficientNetV2-M at 512 px, MSE, SGD (LR 5e-4, momentum .9, Nesterov, weight decay 5e-4), batch 16. The released YAML contains `epochs: 1` but simultaneously fixes cosine `T_max: 25`; this internally inconsistent one-epoch value is treated as a packaging/debug residue and the complete released scheduler horizon (25 epochs) is used. CAM has three channels per each of five grades, matching the reported parameter count. BFFN freezes the corresponding same-seed encoder and uses the released 2560→1280→pool→320→pool→1 head with dropout .3. BFFN's official CLI defaults (2 epochs, batch 64, LR .02) are used unchanged.

Messidor-2 uses the common deterministic inner validation solely for released validation-best selection. This ambiguity and resolution are retained here rather than silently altered.
