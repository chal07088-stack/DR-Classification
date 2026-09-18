# Huang reproduction lock

Source lock: official repository commit `e6cd6b49a27612166f7e60f68bf511ac5030140b` and its `configs/eyepacs.yaml` / `fundus/fusion.py`.

The single model is ImageNet ResNet-50 scalar regression at 512 px, MSE, SGD (LR 1e-3, momentum .9, Nesterov, weight decay 5e-4), cosine decay for 25 epochs and batch 16. Fundus black-border cropping and the released geometric/colour augmentation are applied. PFF freezes the corresponding same-seed selected single encoder and uses the released 4096→2048→pool→512→pool→1 head, SGD LR .02, batch 64, 50 epochs. No ensemble is used.

EyePACS selects by validation QWK. Messidor-2 uses a deterministic 20% inner validation drawn only from the fixed outer-train patients because the released programs select a validation-best checkpoint; the outer test is untouched.
