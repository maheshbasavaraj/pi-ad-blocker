# AR Ad Blocker — Product & Commercial Plan

Status: engineering prototype validated on CPU; accuracy path identified. This
document is the build-and-sell plan that the code has to satisfy.

## 1. Problem and product

Wearable AR (Meta Ray-Ban Display, XREAL, Even Realities, and the coming
Android XR wave) puts a camera and a display on a person's face. Every surface
becomes a line of sight: billboards, transit posters, shop signage, digital
screens. The product is a **real-time, on-device visual ad filter**: detect
commercial signage in the camera feed, composite a mask or replacement into the
display pipeline, and do it with no cloud dependency.

The user-visible promise: *"the city without the ads."* Not a browser ad
blocker — a **world ad blocker**.

Why this is a product and not a demo: a filter that paints a black rectangle
over every pedestrian is worse than no filter. The entire engineering budget
goes into (a) precision on ad concepts, (b) mask stability so the overlay does
not strobe, (c) latency low enough that the overlay does not lag the world.

## 2. Measured reality (CPU baseline, 854×480, 32-core laptop)

| stage | measured | note |
|---|---|---|
| Open-vocab teacher (YOLOE-11s-seg, 640) | 2.2–2.9 fps (340–450 ms) | too slow to be runtime |
| Fine-tuned nano seg (YOLO11n-seg, .pt, 640) | 15.2 fps | measured, this build |
| Fine-tuned nano seg (.pt, 480) | 23.5 fps | measured — clears the 20 fps gate |
| Nano seg ONNX FP32, 640 | 19.5 fps pipeline / 30.0 fps raw | measured |
| Render loop, no encoder (150 frames) | 35–39 fps | optical-flow propagation, masks on 70–80% of frames |
| Mask stability, YOLOE-only | consecutive IoU mean 0.804, p10 0.080 | **fails** the gate |
| Mask stability, nano student | IoU mean 0.885–0.912, p10 0.776–0.800 | **passes** |
| Mask area | YOLOE 1.5–2.8% → nano 10.3–12.2% (p95 22–28%) | bigger masks, more recall; unverified by eye |

Hard-won conclusion: **mask stability is bounded by detector rate, not by masker
tuning.** A 2.3 fps detector leaves ~13 render frames of optical-flow coasting
between keyframes; drift accumulates and snaps back at every new detection. Two
masker-side fixes (cross-class dedupe, hysteresis hold with fade, miss tolerance,
RANSAC partial-affine flow) moved mean IoU 0.662 → 0.804 and left p10 flat at
0.08. Also measured: raising `max_propagation_frames` 15 → 60 makes jitter
*worse*; YOLOE at `imgsz=480` doubles detector fps and detects **nothing** — 640
is its floor.

Therefore the shipping architecture is **teacher → student distillation**:
YOLOE labels the domain (zero human labelling) and a nano segmentation model is
fine-tuned on those labels; measured 15.2 fps @640 and 23.5 fps @480 in-pipeline
on this CPU, higher on an NPU.
Once detection is per-frame, propagation is an optimization rather than a crutch.

## 3. Hardware targets and the performance budget

Meta Ray-Ban Display-class glasses: ~1 camera stream, waveguide display, a
low-power SoC with a small NPU (single-digit TOPS). Nothing open-vocab runs
there. A 3.3 MB dynamic-INT8 export exists but runs **slower** than FP32 (225 ms vs
33 ms); static QDQ calibration is required before any INT8 claim.

Budget at 30 fps display refresh (33 ms per frame):

| task | budget | model |
|---|---|---|
| capture + colour convert | 2 ms | platform |
| detect (per frame or every 2nd) | 8–15 ms | YOLO11n-seg INT8 @480 on NPU |
| track + polygon smoothing | 2 ms | IoU tracker + per-ID EMA |
| mask composite (ROI-bounded) | 3–6 ms | measured 6.4 ms black @854×480 CPU |
| display submit | 2 ms | platform |
| **headroom** | ~8 ms | jitter absorption |

Fallback tier (Raspberry Pi 5 / phone CPU): detector every 2nd frame at 480,
optical-flow propagation in between, ROI-bounded compositing — measured 29.97 fps
paced render on laptop CPU with the slow teacher, so a nano model leaves margin.

## 4. Accuracy strategy

1. **Class precision first.** COCO has no billboard class (its only ad-ish id is
   62 `tv`); a COCO seg model masks 0% of street signage. Open-vocab prompts are
   required for bootstrapping and are unshippable at runtime.
2. **Auto-label then distill.** YOLOE with concrete prompts (`billboard`,
   `advertisement poster`, `store sign`, `digital screen`) labels domain frames;
   a nano seg model learns the domain. Measured label yield: 891 usable frames
   from 977 sampled frames of street footage (91%).
3. **Where open-vocab fails, add OCR as a second detector.** Text-dense surfaces
   (posters, digital screens) are found by EasyOCR even when the object detector
   is unsure; merge with centroid clustering, run async every N frames.
4. **Never over-mask.** Class census is a gate: a run that masks pedestrians,
   cars or traffic lights is a failure regardless of FPS.

## 5. Product surface

- **Glasses filter** (primary): modes `black`, `blur`, `pixelate`, and
  `overlay` (masked region on transparent canvas — what an AR compositor wants).
- **Replacement mode** (upsell): perspective-warped infill so signage is replaced
  by neutral art instead of blacked out; same pipeline, homography + feather.
- **Selective allow-list**: users can whitelist a venue's own signage.
- **No cloud, no accounts required.** Frames never leave the device. This is both
  the privacy story and the latency story.

## 6. Commercial model

| stream | who pays | why |
|---|---|---|
| One-time app unlock ($9–19) | consumer | simple, matches "sunglasses accessory" pricing |
| Subscription ($2.99/mo) | heavy users | replacement art packs, allow-lists, multi-device sync |
| Venue/enterprise licence | museums, clinics, schools, transit operators | remove ads from wayfinding; custom detection categories |
| SDK licence | glasses OEMs | ship the filter as a system feature; per-device royalty |

OEM is the real business: the filter is a *system-level* feature, not an app on
a 600×600 waveguide. A per-device royalty on a 1M-unit OEM line beats consumer
downloads by an order of magnitude, and the moat is the distilled model plus the
stability pipeline, not the app shell.

## 7. Competitive position

Existing ad blocking is display-layer (browsers, DNS). Nothing ships a
world-layer filter, because the problem only became tractable when nano
segmentation models got small enough to run on a glasses NPU. Adjacent players
(Samsung/XREAL/Android XR SDK holders) could ship their own; the defensible part
is domain accuracy on real signage plus the stability work, which is measurable
and slow to copy.

## 8. Risks and mitigations

| risk | mitigation |
|---|---|
| Over-masking (people, cars, signs) | class census gate in CI; single-class distilled model |
| Mask flicker / strobe | p10 consecutive-IoU ≥ 0.75 ship / ≥ 0.85 stretch (measured 0.776–0.800) |
| Detector too slow on device | nano INT8 + per-frame detection; propagation only as fallback |
| Regulatory / venue pushback | personal-device filter, images not stored; enterprise allow-list mode |
| Thermal throttling | 480² input, INT8, detect every 2nd frame under load |
| Model drift across cities/scripts | auto-label pipeline re-runs on new footage; teacher labels, student ships |

## 9. Roadmap

1. ✅ Real-time pipeline: async detector thread + optical-flow warp, polygon masks, AR overlay mode.
2. ✅ Stability work: dedupe, hysteresis hold/fade, miss tolerance, partial-affine flow (measured, insufficient alone).
3. ▶ **Distillation**: YOLOE auto-labels → YOLO11n-seg fine-tune → ONNX INT8 → gate on p10 IoU ≥ 0.85 and detector ≥ 20 fps.
4. OCR second detector for text-heavy surfaces; class census in CI.
5. On-device benchmark harness (phone NPU, RPi 5 + Hailo) with the same gates.
6. Replacement/allow-list modes; OEM SDK surface.

## 10. Kill criteria

Ship-nothing if: p10 consecutive IoU < 0.85 on a held-out street sequence, class
census contains non-ad classes, detector < 20 fps on the reference device, or the
mask area distribution diverges from the sign's real footprint. Every one of
those is measurable with `scripts/mask_quality.py` plus the benchmark harness.
