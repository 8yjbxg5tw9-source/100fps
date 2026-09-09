"""Draw the 100fps app icon (Step 10) — reproducible, no art assets needed.

Dark rounded square + orange play triangle + film-strip perforations.
Run from the repo root::

    python packaging/assets/make_icon.py

Writes ``icon.png`` (1024) and ``icon.ico`` (multi-size) next to itself.
"""

from __future__ import annotations

from pathlib import Path

SIZE = 1024
BG = (18, 20, 28, 255)        # near-black navy
ORANGE = (255, 122, 24, 255)  # brand orange (matches the WebUI progress bar)
ORANGE_DARK = (200, 85, 10, 255)
PERF = (58, 62, 80, 255)      # film perforation grey


def draw_icon(size: int = SIZE):  # noqa: ANN202 - PIL image
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Rounded-square backdrop.
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 6, fill=BG)
    # Film perforations: two vertical hole strips.
    hole_w, hole_h, gap = size * 0.055, size * 0.038, size * 0.062
    for x in (size * 0.055, size * 0.89):
        y = size * 0.06
        while y + hole_h < size * 0.96:
            draw.rounded_rectangle(
                [x, y, x + hole_w, y + hole_h], radius=hole_h // 3, fill=PERF
            )
            y += gap
    # Orange ring + play triangle.
    cx, cy, r = size / 2, size / 2, size * 0.30
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r], outline=ORANGE, width=int(size * 0.045)
    )
    tx = size * 0.115  # triangle half-width tuned to sit inside the ring
    ty = size * 0.15
    draw.polygon(
        [(cx - tx * 0.55, cy - ty), (cx - tx * 0.55, cy + ty),
         (cx + tx, cy)],
        fill=ORANGE,
    )
    # Speed lines under the ring (1000 FPS streaks).
    for i, length in enumerate((0.30, 0.22, 0.14)):
        y = cy + r + size * (0.06 + i * 0.055)
        x0 = cx - size * length / 2
        draw.rounded_rectangle(
            [x0, y, x0 + size * length, y + size * 0.028],
            radius=size * 0.014, fill=ORANGE_DARK,
        )
    return img


def main() -> int:
    here = Path(__file__).resolve().parent
    img = draw_icon()
    img.save(here / "icon.png")
    img.save(
        here / "icon.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
               (128, 128), (256, 256)],
    )
    print(f"wrote {here / 'icon.png'} and {here / 'icon.ico'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
