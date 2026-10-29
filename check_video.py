"""Check video file stats."""
import cv2
from pathlib import Path

for name in ['test_ad_video.mp4', 'masked_output.mp4']:
    p = Path('data/outputs') / name
    if p.exists():
        cap = cv2.VideoCapture(str(p))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps if fps > 0 else 0
        size_mb = p.stat().st_size / (1024*1024)
        cap.release()
        print(f'{name}:')
        print(f'  {w}x{h}, {total} frames @ {fps:.2f} FPS ({duration:.1f}s)')
        print(f'  Size: {size_mb:.1f} MB')
    else:
        print(f'{name}: NOT FOUND')