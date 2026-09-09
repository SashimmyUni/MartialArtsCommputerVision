# Reference Pose Folder Layout (Multi-Angle Ready)

This folder is structured for automatic best-angle matching.

## Technique-first layout

Store each technique in its own folder:

- `reference_poses/<technique>/<angle>.npy`

Examples:

- `reference_poses/front_kick/front.npy`
- `reference_poses/front_kick/left45.npy`
- `reference_poses/front_kick/right45.npy`
- `reference_poses/front_kick/side.npy`

Recommended angle names:

- `front`
- `left45`
- `right45`
- `side`
- `side_right`
- `side_left`
- `behind`

Technique folder names should use snake_case:

- `fighting_stance`, `jab`, `cross`, `hook`, `uppercut`, `front_kick`, `roundhouse_kick`, `side_kick`, `back_kick`, `spinning_back_kick`, `knee_strike`, `elbow_strike`, `axe_kick`

## Karate technique data

| File | Contents |
|---|---|
| `karate_techniques.csv` | Catalogue of 54 karate techniques: romaji, Japanese, English, family, capture profile |
| `karate_capture_plan.csv` | Capture plan generated from the catalogue, one row per (technique, angle) |
| `karate_video_candidates.csv` | Searched candidate source clips per technique, awaiting review |

Karate techniques are stored under their romaji key in the same layout as
everything else — `reference_poses/mae_geri/front.npy`. See
`docs/KARATE_TECHNIQUES.md` for the workflow that fills them.

## Notes

- The active loader supports this nested multi-angle layout directly.
- Existing legacy flat files (for example `front_kick.npy`) can still be loaded when present, but new captures should be stored under technique folders.
