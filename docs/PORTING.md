# Porting the scoring core

This document is for whoever reimplements the trainer's scoring core in another
language — Swift for an iOS app, TypeScript for the browser. It describes the two
artifacts this repo exports for that purpose, and the specific places where a
faithful-looking port silently produces different scores.

Read this before writing any of the port.

## Why the Python is not the thing being ported

The runtime here has three layers, and only one of them travels:

| Layer | Ports? |
|---|---|
| Pose estimation — YOLO26n-pose via Ultralytics/PyTorch | **No.** Replace with a platform detector. |
| Person tracking — Ultralytics ByteTrack, `_select_primary_track` | **No.** A single-user app takes the largest/most-central person. |
| Scoring core — 17 functions, 388 lines, pure NumPy (see the table at the end) | **Yes.** This is the port. |
| Feedback rules — `generate_feedback`, +70 lines | **Yes.** The product needs the coaching strings. |
| Technique classification — `technique_catalog.py`, 152 lines | **Yes.** The scorer depends on it; see below. |
| Reference data — 93 sequences | **Yes**, via the bundle below. |
| UI — `cv2.imshow` plus 57 argparse flags | **No.** Rewritten. |

There is no single contiguous "scoring core" line range. An earlier version of this
document cited `action_recognition.py:726-1352`; that range mixes in ~150 lines of
JSON/CSV run-storage I/O and still omits `generate_feedback` (`:1686`) and
`_build_reference_overlay` (`:1831`). Use the function table at the end of this
document, not a line range.

`technique_catalog.py` is easy to miss and is a hard dependency: `_compare_resampled`
reaches it through `_technique_angle_category` → `technique_family`, which decides
which joint-angle triplets a technique is scored on. It is deliberately stdlib-only
and ports directly. The bundle exports the resolved per-technique category, so a port
that only scores can read that instead of reimplementing the CSV lookup — but a port
with a technique **picker** will want the catalogue itself for the display names.

## The licensing position

This is the constraint that shapes the architecture, so it is worth stating
precisely. **None of this is legal advice; get a lawyer before shipping paid.**

`LICENSE` is the verbatim GNU AGPL v3 and `action_recognition.py:1` carries
`# Ultralytics 🚀 AGPL-3.0 License`. The file is a modified copy of Ultralytics'
own action-recognition example — upstream and local share a character-identical
`crop_and_pad`, the same classifier classes, and the same `run()` defaults. As
published, this program is AGPL-3.0 and cannot go to the App Store.

**But the repo's own LICENSE file is not the binding constraint.** A copyright
holder is not a licensee of their own work: you can relicense what you wrote
whenever you like. What you cannot relicense is Ultralytics' library and their
published weights. So the question is not "is this repo AGPL" but "does the shipped
app contain anything of Ultralytics'".

On the evidence, the scoring algorithms are yours. Grepping the scoring function
names across `ultralytics 8.4.21` returns zero hits, and the upstream example
contains no DTW or pose-scoring code at all. (Ultralytics does ship a pose-angle
solution in `solutions/ai_gym.py`, but this project does not use it.) That is an
**authorship** argument, not an import-graph one — the greps show the code is absent
upstream, and what frees it is that you hold the copyright.

Two things to square away before relying on that:

- Commit `ba9aea40` carries a `Co-Authored-By: Claude Opus 5` trailer, modified
  `action_recognition.py` (+82 lines, including `_technique_angle_category` inside
  the ported range), and created `technique_catalog.py` in its entirety. An AI is
  not a rights-holder, and no human third party appears anywhere in the history of
  these files — every commit touching `action_recognition.py` is authored by the one
  identity — so the relicensing route survives. Re-attest that work as your own
  before relicensing rather than assuming it.
- `yolo26n-pose.pt` sits in the repo root and `action_recognition.py:19-23` links
  five Ultralytics symbols at runtime. Both must be gone from the shipped app.

**What actually conflicts with the App Store.** Not the five-device limit that gets
cited — that figure is a 2010-era iTunes artifact and is not in Apple's current
standard EULA. The durable conflicts are AGPL §10's no-further-restrictions clause
against the EULA's no-transfer / no-redistribute / no-sublicense and
no-reverse-engineer / no-derivative-works terms, plus GPLv3 §3's anti-DRM provision
against FairPlay. The VLC precedent reads the same way: pulled in January 2011 after
a rightsholder complaint, and back on the store on 19 July 2013 once VideoLAN
dual-licensed the iOS app under MPLv2 and relicensed libvlc/VLCKit to LGPLv2.1. The
cure was relicensing — and where VideoLAN needed hundreds of contributors to agree,
this project needs one signature.

**Three consequences worth internalising:**

1. **Exporting the model to CoreML is the worst option, not the safe one.** Bundling
   exported weights into an App Store binary is *distribution* of an
   Ultralytics-derived artifact — a stronger trigger than serving it. Keep the
   export as a parity diagnostic only. (Whether weights are legally a derivative
   work of AGPL training code is contested, not settled.)
2. **A server deployment is the worst licensing position of the three**, which
   inverts the usual intuition. AGPL §13 (`LICENSE:540`) closes the GPLv3 ASP
   loophole: you convey nothing, yet you owe Corresponding Source to every user who
   interacts with the service over a network.
3. **There is a paid escape hatch nobody should ignore.** Ultralytics sells an
   Enterprise licence at the very URL in the file header, expressly for proprietary
   closed-source commercial products. Pricing is quote-only. That turns a stated
   blocker into a build-vs-buy decision and is worth one email before committing to
   replacing the detector.

**Two exposures that are not about software at all.** The reference sequences were
partly derived from scraped YouTube video (see `scripts/scout_youtube_by_golden_seeds.py`
and `reference_poses/karate_video_candidates.csv`), and whether extracted skeletal
keypoints are a derivative work of the source footage is an open question for a paid
product. Separately, 32 of the 54 rows in `reference_poses/karate_techniques.csv`
cite Wikipedia in their `sources` column, which carries CC-BY-SA attribution
obligations if that text ships. Neither is currently owned by anyone.

## The two artifacts

Generate both, then verify them:

```bash
python scripts/export_mobile_bundle.py
```

```bash
python scripts/export_golden_vectors.py
```

```bash
python scripts/verify_export.py
```

They land in `export/mobile/` (gitignored — regenerate rather than commit).

`verify_export.py` is worth running every time, but be clear about what it can and
cannot establish. It loads **only** the exported files — never the source `.npy`
data — rebuilds the references from the blob, rebuilds the inputs from the embedded
lists, re-runs the scoring core, and checks the results against the recorded
expectations. It exits non-zero on failure and works as a CI step.

**What passing proves:** the artifacts are self-contained and internally consistent;
the blob is contiguous, correctly sized, checksum-matching and fully consumed; the
manifest's constants, joint order, skeleton edges, angle triplets and per-technique
categories match the live code; the JSON parses under strict rules; and the recorded
expectations are current rather than stale.

**What passing does NOT prove, and cannot:** that the scoring core is *right*. The
script imports the very functions that generated the vectors, so for the array levels
the comparison is the tautology `f(x) == recorded f(x)`. If `_compare_resampled` has
a design flaw, the exporter records the flaw faithfully and the verifier reproduces it
faithfully. Passing means **bug-compatible with current HEAD** — which is exactly what
a port needs, and is not the same as correct. It also executes no Swift or TypeScript,
tests nothing about model inference or keypoint extraction, and says nothing about
real-time behaviour on a device.

One residual gap worth knowing: for cost matrices above 1024 cells the golden file
records marginals, the diagonal, distribution stats and a float32 SHA-256 rather than
every cell. The marginals alone are not injective — a compensating four-cell
perturbation leaves row sums, column sums, min, max and even the DTW scalar
bit-identical — which is why the diagonal and the checksum were added. With the
checksum in place a wrong cost matrix cannot pass, but note that the checksum pins the
*cost matrix*, not the DTW recurrence; the DTW is pinned separately by
`dtw_min_plus` and by every level-4 score.

It is also the closest thing to a worked example of consuming these files: read it
if the manifest layout or the null-for-NaN convention is unclear.

### 1. `reference_bundle.json` + `reference_bundle.bin`

The reference library, flattened. The manifest carries per-technique/angle byte
offsets into the blob plus everything the scorer would otherwise have to derive
from this repo's Python: the joint-angle triplets per category, the COCO-17 joint
names and skeleton edges, the scoring constants, and the per-label thresholds
when they exist.

The blob is little-endian float32, `(T, 17, 3)` per reference = `(x, y, confidence)`
in **source video pixels, origin top-left, y down**.

Sequences are exported **raw**. Normalization and resampling happen on the
consumer side, deliberately, so both implementations start from identical input
and can be compared at every intermediate step.

Reading one reference, conceptually:

```
entry  = manifest.techniques[technique].references[angle]
floats = blob[entry.offset ..< entry.offset + entry.byte_length] as [Float32]
seq    = floats.reshaped(entry.frames, 17, 3)
```

### 2. `golden_vectors.json`

Expected output of the Python core for a fixed set of inputs, layered by
function. The input windows are embedded as nested lists, so this file is
self-contained — a port needs no `.npy` reader and no Python to run its tests.

| Level | Pins |
|---|---|
| `1_normalize` | `normalize_pose_sequence` |
| `2_resample` | `resample_pose_sequence`, at six target lengths |
| `3_dtw` | full `_pairwise_frame_cost_matrix` **and** `_dtw_min_plus` separately |
| `4_compare` | `compare_pose_sequence` metric bundle, every (fixture, technique, angle) |
| `5_best_match` | `_best_reference_match` chosen angle and metrics, exhaustive and `topk=3` |
| `6_feedback` | `generate_feedback` strings |

The layering is the point. A single end-to-end score mismatch says nothing about
where a port went wrong; these six levels isolate it. Port level by level and get
each one green before starting the next.

Suggested tolerances are in the file's `tolerances` block. Everything is float32
through the pipeline, so exact equality is not a reasonable bar.

## The score formula

Stated here because a port cannot be correct without it, and because one term in it
is genuinely unguessable. From `_compare_resampled` (`action_recognition.py:805-817`):

```
cosine_score    = (cos_sim + 1.0) * 50.0
dtw_score       = max(0, 100 * (1 - dtw_dist   / 0.8))
angle_score     = max(0, 100 * (1 - angle_err  / 90.0))
pose_dist_score = max(0, 100 * (1 - mean_dist  / 0.8))

score = 0.35*cosine_score + 0.25*dtw_score + 0.25*angle_score + 0.15*pose_dist_score
        clipped to [0, 100]
```

Three definitions that are not obvious from the formula:

- **`mean_dist` is the diagonal of the cost matrix** — `np.diagonal(cost).mean()`
  at `action_recognition.py:796` — *not* the mean cost along the DTW warping path.
  It is a straight frame-i-to-frame-i comparison, which is meaningful only because
  both sequences were resampled to the same length. This is the single most likely
  thing for a port to get plausibly wrong.
- **`dtw_dist` is normalized by the matrix dimension** (`:795`), which is why the
  square-matrix property below is load-bearing rather than incidental.
- **`angle_err` defaults to 90.0** when no angle pair is comparable, which drives
  `angle_score` to exactly 0 rather than skipping the term.

The angle triplets come from `_ANGLE_DEF_CATEGORIES` (`:1285`) selected by
`_technique_angle_category` (`:1303`). Both are hard dependencies of the scorer and
both must be ported; the bundle exports the triplets and the per-technique category
so a port need not reimplement the classifier.

## Porting traps

Each of these produces a plausible-looking port with wrong scores.

### NaN matters, but not where you would expect

`normalize_pose_frame` (`action_recognition.py:1138`) sets any joint below
`conf_thresh` to NaN. That much is real, and it is how occluded joints are marked.

But the NaNs **do not survive to the comparison**, and getting this backwards
wastes effort on the wrong function. `resample_pose_sequence`
(`action_recognition.py:1112`) forward-fills, back-fills and linearly interpolates
every NaN away, and writes `0.0` for a joint that is invalid in every frame. Since
`compare_pose_sequence` resamples before building the cost matrix, the input to
`_pairwise_frame_cost_matrix` is always NaN-free — which means its "valid in both
frames" masking and its exact-`1.0` fallback **never fire in this pipeline**.
Verified by construction: forcing a joint's confidence to zero across all 8 frames
yields 16 NaNs after normalization, zero after resampling, and no `1.0` cell in the
cost matrix.

So what a port must reproduce exactly is the **fill behaviour**, not the masking:

- leading NaNs take the first valid value, trailing NaNs take the last
- interior NaNs are linearly interpolated between valid neighbours
- a joint invalid in *every* frame becomes `0.0` — at the body centre after
  normalization, not "absent"

Get the fill wrong and every occluded-joint frame feeds different numbers into an
otherwise-correct comparison. Implementing the cost-matrix validity mask is
harmless but not necessary.

NaN is still load-bearing at level 1 of the golden vectors, where normalized
output legitimately contains it. There, a JSON `null` means NaN, and producing a
number where the golden file has null is a hard failure regardless of tolerance.

### `_mirror_sequence` does not relabel left and right

`action_recognition.py:1349` negates the x coordinate and nothing else. It does
**not** swap left/right joint indices, which is what mirroring a body would
normally mean.

This looks like a bug and a porter will be tempted to fix it. Do not. Every
calibrated threshold and every golden vector assumes the current behaviour, so
"fixing" it silently invalidates all of them. The bundle records this explicitly
as `scoring_constants.mirror_swaps_left_right_indices: false`.

Both orientations are scored and the better cosine similarity wins
(`action_recognition.py:788-791`), so `use_mirror` is an output worth asserting
on, not just an internal detail.

### The DTW cost matrix is always square

`compare_pose_sequence` resamples the user window to the *reference's* own
`target_len` (`action_recognition.py:1664`), and `_prepare_reference_sequence`
resamples the reference to that same length. So `na == nb` for every comparison
the app ever performs, and the cost matrix is always square.

The general min-plus recurrence in `_dtw_min_plus` handles rectangular inputs —
that path is simply never taken. A port can rely on the square case, and its
anti-diagonal bounds are correspondingly simpler. Worth knowing before
reimplementing the index arithmetic in full generality and then debugging it.

The shortest reference in the library is 16 frames and the longest is 196, so
matrix sizes in practice range from 16x16 to 196x196.

### Precision is split — float32 arrays, float64 score algebra

Not "use Float everywhere". The pipeline mixes the two, and a port that picks one
uniformly will not reproduce the golden vectors:

- **Arrays are float32.** Normalized and resampled sequences, the cost matrix, and
  the `_dtw_min_plus` dp table (`action_recognition.py:1226`) are all float32.
  Accumulating the DTW recurrence in double drifts from the recorded values.
- **The score arithmetic is float64.** Every scalar in `_compare_resampled` is a
  Python `float`, i.e. double: the cosine dot product, `dtw_dist`, `angle_err`,
  `mean_dist`, the four sub-scores and the weighted sum. Verified by inspecting the
  returned types.

So: `Float` for the arrays, `Double` for the scalars from `_compare_resampled`
onward. The bundle records this as `array_dtype: float32` and
`score_arithmetic_dtype: float64` rather than a single misleading value.

### Coordinate space — this is where ports actually die

References are in source pixels, origin top-left, y down — OpenCV's convention.

Apple Vision differs on **two** axes at once. It returns normalized `[0,1]`
coordinates with a **bottom-left** origin, so y must be flipped; and it normalizes
x and y **independently by frame width and height**, so the pose is aspect-distorted
until you multiply back by `(frame_width, frame_height)`. Since
`normalize_pose_frame` scales the whole pose by shoulder distance, either mistake
propagates into every metric.

It also returns **19 joints, not 17** — `VNHumanBodyPoseObservation.JointName`
includes `neck` and `root` alongside the COCO set. All 17 COCO joints are present,
so the mapping is total, but it must be written explicitly and in this index order:

| COCO idx | Vision `JointName` | | COCO idx | Vision `JointName` |
|---|---|---|---|---|
| 0 | `nose` | | 9 | `leftWrist` |
| 1 | `leftEye` | | 10 | `rightWrist` |
| 2 | `rightEye` | | 11 | `leftHip` |
| 3 | `leftEar` | | 12 | `rightHip` |
| 4 | `rightEar` | | 13 | `leftKnee` |
| 5 | `leftShoulder` | | 14 | `rightKnee` |
| 6 | `rightShoulder` | | 15 | `leftAnkle` |
| 7 | `leftElbow` | | 16 | `rightAnkle` |
| 8 | `rightElbow` | | — | `neck`, `root` discarded |

Note that Vision's `left`/`right` are **image-side** labels, as COCO's are. Do not
"correct" them to the subject's anatomical left and right.

**Measured cost of getting each of these wrong**, against real reference data —
score drop and resulting top-1 technique identification rate:

| Mistake | Score | Top-1 |
|---|---|---|
| No aspect-ratio restoration | −43 pts | 30% |
| Unflipped bottom-left-origin y | −70 pts | 5% |
| Swapped left/right joint indices | −49 pts | 25% |
| *(for comparison)* 5% shoulder-separation detector bias | −6.2 pts | 100% |
| *(for comparison)* 20%-of-shoulder-width hip offset | −9.4 pts | 100% |

These three deterministic mistakes are an order of magnitude more damaging than the
cross-detector bias everyone worries about. Write a fixture that asserts the
transform against a known pose before writing any scoring code.

TensorFlow.js MoveNet emits COCO-17 directly, in the same index order, so the
mapping is the identity. Its raw output ordering is `[y, x, score]` normalized to a
square letterboxed input — confirm against the current tfjs-models docs before
relying on it.

### `conf_thresh` cuts both ways — do not simply retune it

The `0.2` default is calibrated against YOLO's confidence distribution, and a
different detector's confidences sit on a different scale. The obvious move is to
refit it. **That is a trap.**

The same `conf_thresh` is applied to the *reference* sequences as well as the live
pose: `_prepare_reference_sequence` (`action_recognition.py:729`) passes it into
`normalize_pose_sequence` for the stored YOLO-captured references. Raising it to
suit Apple Vision therefore also re-masks the reference side, discarding
reference joints that were perfectly good, and shifts the baseline it is being
compared against.

If the detector genuinely needs a different gate, the two uses have to be
separated into a live threshold and a reference threshold before either is tuned.
Changing the single shared constant is not a neutral calibration step.

### Reference preparation caching is an implementation detail

`_prepare_reference_sequence` (`action_recognition.py:729`) caches by `id()`
purely as a speed hack, because references are immutable for the run. A port
should just precompute every prepared reference once at bundle load. Nothing
about the cache is semantic.

## Cross-model validation

**Passing every golden vector proves the port computes the same function. It does
not prove the ported app produces the same scores**, because the input changes:
every reference sequence was captured with YOLO26n-pose, and a different detector
places joints differently.

The mechanism is real: `normalize_pose_frame` centres on the hip midpoint and scales
by shoulder distance (`action_recognition.py:1149-1172`), so a systematic offset in
hip or shoulder placement rescales every pose and shifts the score distribution,
invalidating any calibrated threshold.

**But it has now been measured, and it is a recalibratable perturbation rather than a
fundamental invalidation.** Perturbing real reference sequences and re-scoring gives
roughly **1.2–1.3 score points per 1% of shoulder-separation bias** at small biases,
flattening to ~1.0 pt/% by 20%:

| Injected bias | Mean score drop | Top-1 technique ID |
|---|---|---|
| 5% shoulder separation | 6.2 pts | 100% |
| 20%-of-shoulder-width hip shift | 9.4 pts | 100% |
| 5% shoulder + 5% hip + 5% per-joint gaussian noise, together | ~12 pts | 20/20 correct |

Two caveats on those figures. The worst-case sequences are about **3× more sensitive
than the mean** — 5% bias costs 17.2 points on the most fragile sequence — so
per-technique recalibration matters more than a single global shift. And the combined
result is governed almost entirely by the assumed per-joint noise sigma, which nobody
has measured for Apple Vision; at sigma ≈ 0.1 of shoulder width, top-1 degrades to
around 70%. Measuring that sigma is the experiment that actually matters.

So rank the risks accordingly: the deterministic coordinate-convention mistakes in the
section above cost 43–70 points and collapse technique identification, while detector
bias costs single digits and preserves it. Fix the transform first; treat drift as a
threshold-recalibration task.

Validate before trusting the ported scores:

1. Record clips per technique.
2. Score them with the port, and with this repo:
   `python action_recognition.py --source <clip> --target-technique <t> --disable-video-classifier --no-display`,
   which writes `data/runs/<id>/metrics.csv`.
3. Compare distributions per technique. `plot_score_distributions_grouped.py`
   already groups exactly this way and can be pointed at both sets.
4. Recalibrate with `scripts/calibrate_label_thresholds.py` rather than by hand.

Do this for **one** technique early, before the full port is finished. It is the
cheapest available insurance against discovering at the end that the reference
library needs re-capturing.

If the gap cannot be closed by recalibration, the fallback is to export the
original model — Ultralytics 8.4.21 lists CoreML among its export formats, so
`yolo26n-pose.pt` to `.mlpackage` is nominally one command and preserves parity
exactly. Two things to know before relying on it: CoreML export needs `coremltools`,
which is **not** installed in this repo's venv, so the export has not actually been
run here; and shipping the result is *distribution* of an Ultralytics-derived
artifact, which is a stronger licensing trigger than serving it. Keep it as a parity
diagnostic, not a shipping plan.

Even as a diagnostic it is not free: exporting the model gets you raw tensor output,
so you would also have to reimplement pose keypoint decoding, NMS, and letterbox
preprocessing — everything `yolo_model.track()` currently does for you, plus the
ByteTrack layer.

## Functions to port

In dependency order, each line number verified. The transitive closure reachable
from `compare_pose_sequence` / `_best_reference_match` is exactly **17 functions,
388 source lines**, and its only dependencies are NumPy,
`collections.defaultdict`, and `technique_catalog.technique_family` — no torch,
cv2, ultralytics, torchvision or transformers. Adding `generate_feedback` for the
coaching strings brings the numeric surface to ~460 lines.

| Function | Line |
|---|---|
| `_valid_point` | 1074 |
| `_safe_joint` | 1078 |
| `_joint_angle_deg` | 1089 |
| `_resample_1d` | 1101 |
| `resample_pose_sequence` | 1112 |
| `normalize_pose_frame` | 1138 |
| `normalize_pose_sequence` | 1179 |
| `_frame_pose_distance` | 1184 |
| `_pairwise_frame_cost_matrix` | 1191 |
| `_dtw_min_plus` | 1216 |
| `dtw_pose_distance` | 1241 |
| `_mean_angle_sequence` | 1254 |
| `_technique_angle_category` | 1303 |
| `_sequence_mean_angles` | 1322 |
| `_mirror_sequence` | 1349 |
| `_prepare_reference_sequence` | 729 |
| `_compare_resampled` | 764 |
| `_cosine_prescreen` | 821 (only needed if you implement `--score-topk`) |
| `_best_reference_match` | 855 |
| `generate_feedback` | 1686 (+70 lines, needed for the product) |
| `_build_reference_overlay` | 1831 (geometry only; drawing is platform code) |

Plus three module constants — `_REF_PREP_CACHE` (726), `_ANGLE_DEF_CATEGORIES`
(1285), `_FAMILY_ANGLE_CATEGORIES` (1294) — and `COCO17_EDGES` (453) for the
skeleton overlay. The bundle exports the edges and triplets so they need not be
retyped.

`_technique_angle_category` and `_ANGLE_DEF_CATEGORIES` are easy to overlook and are
hard dependencies of `_compare_resampled`: they select which joint-angle triplets a
technique is scored against, i.e. 25% of the final score.

Compute needs a real measurement, not an assumption. The live window is 8 frames
of 17 joints, but it is resampled **up** to each reference's length before
comparison, so the grids are square and large: 16×16 to 196×196, and one pass over
the whole library evaluates **1,160,998 cost cells**. Per scored frame, against one
technique's bank, measured on a desktop CPU: `_best_reference_match` costs
**207.6 ms** for jab's 21 references and **1279.6 ms** across all 93. The current
Python therefore sustains roughly 4–5 scored frames per second — the live-camera
premise has never been demonstrated end to end in any language.

That cost is not intrinsic. Roughly 55% is NumPy temporary allocation (~118 MB per
scored frame for `front_kick`, running near the machine's memory bandwidth) and
~40% is Python-level dispatch in the DTW anti-diagonal loop; neither exists in a
fused Swift loop. A scalar Swift port of the same square algorithm is estimated at
**10–20 ms** for the worst technique on a recent A-series core, and proportionally
worse on A12/A13-class hardware — viable at 30fps, but without comfortable margin
once pose inference and rendering are counted. Treat Accelerate/SIMD as likely
needed on older devices rather than a later optimization, and benchmark before
committing to scoring every frame (`--score-every` exists precisely for this).

## Known issue carried by the data

`uppercut` and `elbow` export with `angle_category: "other"`, which scores them
against **knee** angles rather than the shoulder-torso angles used for punches.
Both are arm techniques, so this is wrong; it happens because
`_technique_angle_category` (`action_recognition.py:1303`) matches only "jab",
"cross", "hook" and "kick" by keyword and neither name is in the karate
catalogue.

It is left as-is deliberately: fixing it changes scores for those two techniques
and would invalidate the golden vectors mid-port. A port should reproduce the
exported categories, and the fix should be made in the Python first, with the
vectors regenerated afterwards.
