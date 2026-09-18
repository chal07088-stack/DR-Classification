# D-ODR reproduction lock

Source lock: official repository commit `697dc509625b395d08dbe0804430fd903270fbf5` and MICCAI/main paper configuration. ViT-B/16, 224 px, ImageNet normalization, horizontal flip, physical batch 128, AdamW LR 1e-4, weight decay 1e-4, 100 epochs, FP16, `lambda=1`, `k=25`, scales 1/2/3. The released validation selector `(QWK+1)/2 + Macro-F1` is retained. Fellow eyes are metadata only and never enter this model. Gradient accumulation is fixed to one because the regularizer is minibatch relational.
