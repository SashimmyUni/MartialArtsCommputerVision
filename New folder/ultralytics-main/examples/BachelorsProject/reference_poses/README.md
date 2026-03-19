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

## Notes

- Existing legacy flat files (for example `front_kick.npy`) were kept for backward compatibility.
- Current loader logic in the script still expects flat files by default; this folder structure is prepared for the upcoming automatic best-angle matching logic.
