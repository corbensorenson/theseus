"""Generate the Project Theseus Windows icon without external dependencies."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "windows" / "theseus-hive.ico"
SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Project Theseus Hive Windows .ico asset.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images = [(size, dib_for_size(size)) for size in SIZES]
    out.write_bytes(icon_bytes(images))
    print(str(out))
    return 0


def icon_bytes(images: list[tuple[int, bytes]]) -> bytes:
    header = struct.pack("<HHH", 0, 1, len(images))
    entries = []
    payloads = []
    offset = 6 + 16 * len(images)
    for size, payload in images:
        width = 0 if size >= 256 else size
        height = 0 if size >= 256 else size
        entries.append(struct.pack("<BBBBHHII", width, height, 0, 0, 1, 32, len(payload), offset))
        payloads.append(payload)
        offset += len(payload)
    return header + b"".join(entries) + b"".join(payloads)


def dib_for_size(size: int) -> bytes:
    pixels = render_rgba(size)
    xor = bytearray()
    for y in range(size - 1, -1, -1):
        for x in range(size):
            r, g, b, a = pixels[y][x]
            xor.extend((b, g, r, a))
    mask_stride = ((size + 31) // 32) * 4
    and_mask = bytes(mask_stride * size)
    header = struct.pack(
        "<IIIHHIIIIII",
        40,
        size,
        size * 2,
        1,
        32,
        0,
        len(xor) + len(and_mask),
        0,
        0,
        0,
        0,
    )
    return header + bytes(xor) + and_mask


def render_rgba(size: int) -> list[list[tuple[int, int, int, int]]]:
    canvas: list[list[tuple[int, int, int, int]]] = []
    for y in range(size):
        row = []
        for x in range(size):
            nx = (x + 0.5) / size * 2.0 - 1.0
            ny = (y + 0.5) / size * 2.0 - 1.0
            radius = math.hypot(nx, ny)
            edge = smoothstep(0.98, 0.9, radius)
            if edge <= 0.0:
                row.append((0, 0, 0, 0))
                continue
            shade = 1.0 - min(1.0, radius)
            base = blend_rgb((8, 14, 26), (13, 68, 92), shade * 0.85)
            row.append((*base, int(255 * edge)))
        canvas.append(row)

    draw_orbit(canvas, size, angle=0.58, rx=0.88, ry=0.35, color=(90, 222, 232), width=0.035, alpha=0.72)
    draw_orbit(canvas, size, angle=-0.72, rx=0.78, ry=0.3, color=(164, 122, 255), width=0.03, alpha=0.58)
    draw_disc(canvas, size, cx=0.47, cy=-0.42, radius=0.075, color=(247, 246, 210), alpha=0.95)
    draw_disc(canvas, size, cx=-0.58, cy=0.28, radius=0.045, color=(94, 231, 175), alpha=0.82)
    draw_t_mark(canvas, size)
    return canvas


def draw_t_mark(canvas: list[list[tuple[int, int, int, int]]], size: int) -> None:
    draw_rect(canvas, size, -0.48, -0.34, 0.48, -0.12, (242, 248, 252), 0.96)
    draw_rect(canvas, size, -0.13, -0.28, 0.13, 0.53, (242, 248, 252), 0.96)
    draw_rect(canvas, size, -0.34, 0.43, 0.34, 0.63, (242, 248, 252), 0.92)
    draw_rect(canvas, size, -0.1, -0.1, 0.1, 0.48, (96, 231, 220), 0.18)


def draw_rect(
    canvas: list[list[tuple[int, int, int, int]]],
    size: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    feather = 2.2 / size
    for y in range(size):
        ny = (y + 0.5) / size * 2.0 - 1.0
        for x in range(size):
            nx = (x + 0.5) / size * 2.0 - 1.0
            dx = max(x0 - nx, 0.0, nx - x1)
            dy = max(y0 - ny, 0.0, ny - y1)
            coverage = smoothstep(feather, 0.0, math.hypot(dx, dy))
            if coverage > 0.0:
                put(canvas, x, y, color, alpha * coverage)


def draw_disc(
    canvas: list[list[tuple[int, int, int, int]]],
    size: int,
    cx: float,
    cy: float,
    radius: float,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    feather = 2.0 / size
    for y in range(size):
        ny = (y + 0.5) / size * 2.0 - 1.0
        for x in range(size):
            nx = (x + 0.5) / size * 2.0 - 1.0
            coverage = smoothstep(radius + feather, radius - feather, math.hypot(nx - cx, ny - cy))
            if coverage > 0.0:
                put(canvas, x, y, color, alpha * coverage)


def draw_orbit(
    canvas: list[list[tuple[int, int, int, int]]],
    size: int,
    *,
    angle: float,
    rx: float,
    ry: float,
    color: tuple[int, int, int],
    width: float,
    alpha: float,
) -> None:
    ca = math.cos(angle)
    sa = math.sin(angle)
    feather = 2.0 / size
    for y in range(size):
        ny = (y + 0.5) / size * 2.0 - 1.0
        for x in range(size):
            nx = (x + 0.5) / size * 2.0 - 1.0
            xr = nx * ca - ny * sa
            yr = nx * sa + ny * ca
            ellipse = math.hypot(xr / rx, yr / ry)
            dist = abs(ellipse - 1.0) * min(rx, ry)
            coverage = smoothstep(width + feather, width - feather, dist)
            clip = smoothstep(0.98, 0.9, math.hypot(nx, ny))
            if coverage > 0.0 and clip > 0.0:
                put(canvas, x, y, color, alpha * coverage * clip)


def put(canvas: list[list[tuple[int, int, int, int]]], x: int, y: int, color: tuple[int, int, int], alpha: float) -> None:
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0:
        return
    sr, sg, sb = color
    dr, dg, db, da = canvas[y][x]
    sa = alpha
    da_f = da / 255.0
    out_a = sa + da_f * (1.0 - sa)
    if out_a <= 0.0:
        canvas[y][x] = (0, 0, 0, 0)
        return
    out_r = (sr * sa + dr * da_f * (1.0 - sa)) / out_a
    out_g = (sg * sa + dg * da_f * (1.0 - sa)) / out_a
    out_b = (sb * sa + db * da_f * (1.0 - sa)) / out_a
    canvas[y][x] = (int(out_r), int(out_g), int(out_b), int(255 * out_a))


def blend_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


if __name__ == "__main__":
    raise SystemExit(main())
