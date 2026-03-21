# BachelorsProject

Reference-based martial arts coaching project built on top of Ultralytics pose tracking.

## What this folder contains

- `action_recognition.py`: live trainer, reference capture, and evaluation runtime
- `run_reference_collection_batch.py`: batch capture runner from the CSV plan
- `run_golden_seed_technique.py`: extract references from local Golden Seeds videos
- `reference_poses/`: active reference library, capture plan, scout outputs, and previews
- `ReadyToRunCommands.md`: current command snippets for common workflows
- `HOWTO.md`: project handover and architecture notes

## Reference layout

The active reference library uses technique-first snake_case folders:

- `reference_poses/jab/`
- `reference_poses/front_kick/`
- `reference_poses/fighting_stance/`
- `reference_poses/knee_strike/`

Golden Seeds source videos remain under PascalCase folders inside `reference_poses/Golden_Seeds/` because the scouting and extraction scripts map those source folders into snake_case technique keys at save time.

## Quick start

From the repository root on Windows PowerShell:

```powershell
& ".\.venv\Scripts\Activate.ps1"
cd ".\New folder\ultralytics-main\examples\BachelorsProject"
python action_recognition.py --source 0 --target-technique jab --reference-dir reference_poses
```

## Related docs

- `ReadyToRunCommands.md` for current runnable commands
- `HOWTO.md` for the full handover guide
- `reference_poses/README.md` for reference storage conventions
