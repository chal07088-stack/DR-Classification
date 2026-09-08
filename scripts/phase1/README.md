# PHASE 1 Scripts

Run these scripts from PowerShell.

```powershell
.\scripts\phase1\run_vram_probe.ps1
.\scripts\phase1\run_phase1_smoke.ps1
.\scripts\phase1\run_phase1_seed42.ps1
```

`run_phase1_seed42.ps1` is the main local experiment entry point. It runs only
the PHASE 1 scope: ConvNeXt-S, seed 42, EyePACS, 5-fold patient-level OOF,
Teacher A/B, boundary-wise gate, and A-G ablation.
