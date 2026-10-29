"""Create a neutral infill placeholder for the replacement masker.

Generates a subtle wall/sky-colored gradient that blends naturally
when warped onto detected screen regions — no ad content, just
a blank seamless fill.
"""

import cv2
import numpy as np
from pathlib import Path


def create_neutral_infill(
    width: int = 640,
    height: int = 360,
    output_path: str = "data/natural_infill.png",
) -> None:
    """Create a neutral wall-colored gradient for natural infill.

    Uses soft beige/warm-gray tones that blend well with indoor
    and outdoor environments. Subtle gradient adds depth so it
    doesn't look like a flat solid rectangle.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Warm neutral palette: soft beige → light warm gray
    for y in range(height):
        ratio = y / height
        r = int(200 + ratio * 25)    # 200→225 red
        g = int(185 + ratio * 20)    # 185→205 green
        b = int(170 + ratio * 15)    # 170→185 blue
        # Add subtle horizontal variation
        for x in range(width):
            x_var = np.sin(x * 0.02) * 3
            img[y, x] = (
                min(255, max(0, b + int(x_var))),
                min(255, max(0, g + int(x_var))),
                min(255, max(0, r + int(x_var))),
            )

    # Very subtle texture (barely visible noise for realism)
    noise = np.random.randint(-4, 5, (height, width, 3), dtype=np.int8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), img)
    print(f"Created neutral infill: {output} ({width}x{height})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", default="data/natural_infill.png")
    args = parser.parse_args()

    create_neutral_infill(args.width, args.height, args.output)