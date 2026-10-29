"""Verify masking is happening by checking frames throughout the video."""
import cv2
import numpy as np

inp = cv2.VideoCapture('data/outputs/test_ad_video.mp4')
out = cv2.VideoCapture('data/outputs/masked_output.mp4')
total = min(int(inp.get(cv2.CAP_PROP_FRAME_COUNT)), int(out.get(cv2.CAP_PROP_FRAME_COUNT)))

checked = 0
masked_frames = 0
for frame_idx in range(0, total, 500):
    inp.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    out.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret_i, fi = inp.read()
    ret_o, fo = out.read()
    if not ret_i or not ret_o:
        continue
    
    checked += 1
    diff = cv2.absdiff(fi, fo)
    changed = np.sum(diff > 0)
    if changed > 0:
        masked_frames += 1
        pct = changed / diff.size * 100
        print(f'Frame {frame_idx}: ✅ MASKED — {changed} pixels ({pct:.1f}%)')
    else:
        print(f'Frame {frame_idx}: ⚠️  No masking')

inp.release()
out.release()
print(f'\nSummary: {masked_frames}/{checked} sampled frames had masked regions')