# Documentation

| File | Purpose |
|---|---|
| `HOWTO.md` | Primary handover guide — architecture, scoring model, capture gates, CLI reference |
| `ReadyToRunCommands.md` | Copy-paste command cookbook for common workflows |
| `KARATE_TECHNIQUES.md` | The karate technique catalogue — naming, sources, and how to capture references for it |
| `GOLDEN_SEEDS_SCOUT_GUIDE.md` | Walkthrough for sourcing reference clips via the YouTube scout |
| `SCOUT_ARCHITECTURE.md` | Design notes for the scout scripts and their CSV data flow |
| `PORTING.md` | Reimplementing the scoring core in another language (iOS/web) — exported artifacts and the traps |
| `figures/` | Rendered metric plots kept for the report |

## Reference layout

The reference library uses technique-first snake_case folders under
`reference_poses/`:

- `reference_poses/jab/`
- `reference_poses/front_kick/`
- `reference_poses/fighting_stance/`
- `reference_poses/knee_kick/`

Karate techniques use the same layout under their romaji key
(`reference_poses/mae_geri/`, `reference_poses/gyaku_zuki/`); the vocabulary is
catalogued in `reference_poses/karate_techniques.csv` and documented in
`KARATE_TECHNIQUES.md`.

Each holds `<angle>.npy` files of shape `(T, 17, 3)` — COCO-17 keypoints as
`(x, y, confidence)` in raw pixel coordinates. Normalization happens at compare
time, not at capture time.

See `reference_poses/README.md` for the storage conventions.

## Quick start

From the repository root:

```powershell
& ".\.venv\Scripts\Activate.ps1"
```

```powershell
python action_recognition.py --source 0 --target-technique jab
```
