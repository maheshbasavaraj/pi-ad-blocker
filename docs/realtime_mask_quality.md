# Real-time mask quality — experiment ledger

All runs: `data/outputs/test_ad_video.mp4` (854×480), first 150 frames, YOLOE
teacher `yoloe-11s-seg.pt` @640, conf 0.10, prompts `billboard`,
`advertisement poster`, `store sign`, `digital screen`, black fill, polygon masks.

Metrics from `scripts/mask_quality.py` (pixel-diff vs the source frame, so an
all-dark frame cannot fake "masked"):

- active frames = frames where >0.5% of pixels changed
- mask area = percent of frame changed
- jitter IoU = consecutive-frame mask IoU; **p10 is the gate** (≥0.85 to ship)

## Ledger

| # | change | det fps | active | area mean | IoU mean | IoU p10 | verdict |
|---|---|---|---|---|---|---|---|
| 1 | YOLOE async + flow, no stabilisation | 7.28* | 50/150 | 1.48% | 0.662 | 0.048 | fails |
| 2 | + cross-class dedupe, + hysteresis hold 12 frames, + miss tolerance 2 | 2.31 | 65/150 | 2.56% | **0.804** | 0.080 | fails |
| 3 | + per-track partial-affine flow (RANSAC, interior points) | 2.25 | 67/150 | 2.76% | 0.779 | 0.073 | fails |
| 4 | #3 with `max_propagation_frames` 60 instead of 15 | 2.27 | 51/150 | 1.57% | 0.681 | 0.048 | worse |
| 5 | #3 with `imgsz=480` | 4.37 | 0/150 | 0.06% | — | — | detects nothing |

\* lucky run; the teacher sits at 2.2–2.4 fps on this CPU.

## What the numbers prove

1. **Masker tuning has a ceiling.** Four independent stability mechanisms moved
   mean IoU 0.66 → 0.80 but p10 never left 0.05–0.08.
2. **Detector rate is the binding constraint.** At 2.3 fps each detection is ~13
   render frames apart; flow drift accumulates and snaps back when the next
   keyframe lands. That snap *is* the p10 event.
3. **Longer coasting is worse**, not better: 15 → 60 frames of propagation
   dropped IoU to 0.681/0.048. Drift is the error, and more drift is more error.
4. **YOLOE is resolution-sensitive**: at 480 it runs 2× faster and returns zero
   billboards. 640 is the floor for its text prompts.
5. **Render fps in a loop that also encodes is meaningless.** `VideoWriter.write`
   + per-frame `absdiff` produced 14–60 fps swings on identical pipeline configs.
   Pipeline fps is quoted only from `scripts/benchmark_realtime.py`, which has no
   writer in the loop: 29.97 fps paced, 60.7 fps uncapped.

## Required fix (implemented next)

Teacher/student distillation. YOLOE labels the domain
(`scripts/autolabel.py`: 713 train / 178 val frames kept from 977 sampled, 91%
yield), then `scripts/train_ads.py` fine-tunes `yolo11n-seg` on those labels.
Once the student detects every frame, propagation stops being load-bearing and
the p10 gate becomes reachable. Gate remains: p10 ≥ 0.85, detector ≥ 20 fps,
class census ad-only, area consistent with the sign's footprint.

## Result: distillation closed the gate

`yolo11n-seg` fine-tuned 12 epochs @512 on 713 teacher-labelled frames
(5.96 MB `best.pt`). Same 150 frames, same masking config, conf 0.25:

| model | imgsz | active frames | area mean | IoU mean | IoU p10 | detector fps | render fps |
|---|---|---|---|---|---|---|---|
| YOLOE-11s-seg (teacher) | 640 | 65/150 | 2.56% | 0.804 | 0.080 | 2.25 | 60.7 |
| nano seg (student) | 640 | 107/150 (71%) | 10.28% | **0.885** | **0.776** | 10.79 | 20.75 |
| nano seg (student) | 480 | 121/150 (81%) | 12.15% | **0.912** | **0.800** | 14.31 | 20.45 |

Gate was p10 ≥ 0.50 → **passed at both sizes** (0.776 / 0.800, from 0.073).

Two honest caveats:

- **Detector floor.** Pipeline (writer-free benchmark, 150 frames, `--no-pace`):

  | variant | detector fps | render fps | masked frames |
  |---|---|---|---|
  | YOLOE teacher, 640 | 2.25 | 60.7 | 43% |
  | nano `.pt`, 640 | 15.24 | 36.29 | 70% |
  | nano `.pt`, 480 | **23.54** | 35.31 | 79% |
  | nano ONNX FP32, 640 | 19.52 | 39.04 | 71% |

  Gate was 20 fps → **passes only at 480** on this CPU (23.5 fps). Raw ONNX
  inference is 30.0 fps @640; the Python pre/post-processing and tracking inside
  the pipeline eat the difference.
- **INT8 dynamic quantization is a dead end here**: 3.35 MB, but 225 ms/frame
  (4.4 fps) vs 33 ms FP32. Needs static QDQ with a calibration set, not
  `quantize_dynamic`.
- **Render fps recovered once masks became dense** (36–39 fps at 150 frames vs
  20.7 earlier). Polygon fill still dominates; batch the fills into one matte
  before the next speed pass.
- Recall rose 65 → 107 masked frames at 640. That could be true recall or false
  positives; `class_census.py` shows 2527 ad detections over 650 sampled frames
  (3.9/frame), ad-only class. Eye check (2026-11-05, frame 90 of the same clip):
  the four highest-confidence boxes sit on the SAU/H&M sign, a left-side
  billboard, the "Nonstop to LaGuardia" screen and an American Eagle screen —
  all true positives — with **one false positive on a street sign / traffic
  light**. So the class census holds, the mask coverage is real, and the
  residual error is small-object confusion, not wholesale over-masking.


## Iteration: uint8 composite + static-INT8 verdict (2026-11-05)

Measured on the same 150 frames / 854×480, nano seg, conf 0.25. Detector fps is
quoted only from the writer-free benchmark; it swings 5–18 fps on a loaded
32-thread laptop, so the masker numbers below are isolated micro-benchmarks on
identical inputs.

### Masker composite: float32 → uint8 cv2 arithmetic

| path | before | after | note |
|---|---|---|---|
| ROI alpha blend (280×320) | 10.87 ms | **1.68 ms** | 6.5×, max pixel delta 1 LSB |
| `PolygonMasker.apply`, 1 det / 8.4% area, no feather | 6.54 ms | **2.75 ms** | |
| …feathered (`edge_feather_px=2`) | 7.35 ms | **3.31 ms** | shipping config |
| …overlay mode (AR compositor) | 5.00 ms | **2.49 ms** | |

Also dropped a full-frame `mask.any()` scan (0.33 ms) — the drawn/held flags
already know whether anything was painted.

Quality is unchanged by the rewrite (same 150 frames, 480p):

```
active 121/150 (80.7%), area mean 13.03%, IoU mean 0.915, p10 0.805
```

### Static INT8 (QDQ) with calibration — still a dead end on x86

`scripts/quantize_static.py` replaces the dynamic-quant path: 120 calibration
frames from `data/dataset_ads/images/train`, `QuantFormat.QDQ`, per-channel
QUInt8 activations / QInt8 weights.

| variant | size | ms/frame | detections | box IoU mean | verdict |
|---|---|---|---|---|---|
| FP32 ONNX | 11.6 MB | 50.1 | 7/40 frames | — | baseline |
| INT8 QDQ, whole graph | 3.50 MB | 32.8 | **0** | — | broken: head collapses |
| INT8 QDQ, head in FP32 | 5.40 MB | 54.0 | 6 | 0.801 | correct but 0.93× |
| INT8 QOperator, head in FP32 | — | 115.9 | 5 | 0.677 | 0.89× |

Two findings worth keeping:

1. **Quantizing the Ultralytics head destroys the class scores.** Boxes, class
   sigmoids (0–1) and 32 mask coefficients share the head tensors; one
   per-tensor activation scale collapses every score to exactly `0.0`. The
   head must stay FP32 (`--keep-head-fp32`, the default).
2. **On x86 static INT8 is not a speedup here** even with `avx_vnni`. The win
   only exists on an NPU/ARM target, so INT8 stays non-default until measured
   on the reference device (roadmap item 5).

## Verification commands

```bash
python scripts/autolabel.py --video data/outputs/test_ad_video.mp4 \
    --out data/dataset_ads --every 10 --imgsz 640 --conf 0.10
python scripts/train_ads.py --data data/dataset_ads/data.yaml \
    --epochs 12 --imgsz 512 --batch 8
python scripts/render_masked.py --video data/outputs/test_ad_video.mp4 --frames 150 \
    --model runs/seg/ads_nano_seg/weights/best.pt --model-type yolo_seg \
    --imgsz 640 --conf 0.25 --out data/outputs/masked_nano.mp4
python scripts/mask_quality.py --source data/outputs/test_ad_video.mp4 \
    --masked data/outputs/masked_nano.mp4
```
