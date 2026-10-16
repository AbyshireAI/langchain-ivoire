"""Build docs/assets/bla_di_like.stl, the 3D waveform in the repository README.

The model is the amplitude envelope of the phrasebook recording "Bla di like" ("Come and eat"),
spoken by the project's Baoulé voice: N RMS bars extruded as boxes on a base plate, written as
ASCII STL (GitHub renders ```stl blocks and .stl files with an interactive viewer).

    python tools/make_readme_stl.py PATH/32_come_and_eat.wav --readme README.md

Standard library only. The WAV is read, never modified. Every output is re-parsed by ``check``:
facet syntax, solid/endsolid names, and every normal agreeing with its triangle's winding.
"""

from __future__ import annotations

import argparse
import array
import math
import re
import sys
import wave
from pathlib import Path

Vec = tuple[float, float, float]
Tri = tuple[Vec, Vec, Vec]

BARS = 48
BAR_W, GAP = 1.0, 0.45  # mm-ish units; the viewer auto-scales
DEPTH = 4.0
MAX_H, MIN_H = 18.0, 0.6
PLATE_PAD, PLATE_T = 1.5, 1.2


def envelope(path: Path, bars: int = BARS) -> list[float]:
    """RMS of ``bars`` equal slices of a 16-bit PCM WAV (channels averaged), scaled to 0..1."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("expected 16-bit PCM")
        ch = w.getnchannels()
        pcm = array.array("h", w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        pcm.byteswap()
    mono = [sum(pcm[i : i + ch]) / ch for i in range(0, len(pcm), ch)]
    n = len(mono) // bars
    rms = [math.sqrt(sum(x * x for x in mono[k * n : (k + 1) * n]) / n) for k in range(bars)]
    top = max(rms) or 1.0
    return [r / top for r in rms]


def box(
    x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, *, bottom: bool = True
) -> list[Tri]:
    """Outward-facing triangles (counter-clockwise seen from outside): 12, or 10 without the bottom
    face (a bar standing on the plate, whose bottom is hidden)."""
    p = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]  # fmt: skip
    quads = [
        (0, 3, 2, 1),  # bottom (-z)
        (4, 5, 6, 7),  # top (+z)
        (0, 1, 5, 4),  # front (-y)
        (2, 3, 7, 6),  # back (+y)
        (1, 2, 6, 5),  # right (+x)
        (3, 0, 4, 7),  # left (-x)
    ]
    tris: list[Tri] = []
    for a, b, c, d in quads if bottom else quads[1:]:
        tris += [(p[a], p[b], p[c]), (p[a], p[c], p[d])]
    return tris


def normal(t: Tri) -> Vec:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    m = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / m, ny / m, nz / m)


def model(env: list[float]) -> list[Tri]:
    pitch = BAR_W + GAP
    width = len(env) * pitch - GAP
    tris = box(-PLATE_PAD, -PLATE_PAD, 0.0, width + PLATE_PAD, DEPTH + PLATE_PAD, PLATE_T)
    for k, e in enumerate(env):
        h = MIN_H + (MAX_H - MIN_H) * e
        x = k * pitch
        tris += box(x, 0.0, PLATE_T, x + BAR_W, DEPTH, PLATE_T + h, bottom=False)
    return tris


def to_ascii_stl(tris: list[Tri], name: str, *, compact: bool = False) -> str:
    """ASCII STL. ``compact`` puts each facet on one line (the same tokens; STL is
    whitespace-agnostic), which keeps the README's ```stl block short."""
    f = "{:.3f}".format
    out = [f"solid {name}"]
    for t in tris:
        n = normal(t)
        head = f"facet normal {f(n[0])} {f(n[1])} {f(n[2])}"
        verts = [f"vertex {f(v[0])} {f(v[1])} {f(v[2])}" for v in t]
        if compact:
            out.append(" ".join([head, "outer loop", *verts, "endloop", "endfacet"]))
        else:
            out += ["  " + head, "    outer loop", *("      " + v for v in verts)]
            out += ["    endloop", "  endfacet"]
    out.append(f"endsolid {name}")
    return "\n".join(out) + "\n"


_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_FACET = re.compile(
    r"\s*facet normal " + " ".join([_NUM] * 3) + r"\s+outer loop"
    + "".join(r"\s+vertex " + " ".join([_NUM] * 3) for _ in range(3))
    + r"\s+endloop\s+endfacet"
)  # fmt: skip


def check(text: str) -> int:
    """Parse ASCII STL strictly; return the facet count or raise ValueError."""
    lines = text.strip().splitlines()
    m0, m1 = re.fullmatch(r"solid (\S+)", lines[0]), re.fullmatch(r"endsolid (\S+)", lines[-1])
    if not (m0 and m1 and m0[1] == m1[1]):
        raise ValueError("bad solid/endsolid")
    body = "\n".join(lines[1:-1])
    facets = list(_FACET.finditer(body))
    if "".join(m[0] for m in facets).split() != body.split():
        raise ValueError("unparsed text between facets")
    for m in facets:
        v = [float(x) for x in m.groups()]
        n, tri = v[:3], (tuple(v[3:6]), tuple(v[6:9]), tuple(v[9:12]))
        c = normal(tri)  # type: ignore[arg-type]
        if sum(a * b for a, b in zip(n, c, strict=True)) < 0.99:
            raise ValueError("normal disagrees with vertex winding")
    return len(facets)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wav", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/assets/bla_di_like.stl"))
    ap.add_argument("--bars", type=int, default=BARS)
    ap.add_argument(
        "--readme", type=Path, help="also replace the ```stl block in this Markdown file"
    )
    a = ap.parse_args()
    tris = model(envelope(a.wav, a.bars))
    text = to_ascii_stl(tris, "bla_di_like")
    n = check(text)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="ascii", newline="\n")
    print(f"{a.out}: {n} facets, {len(text) / 1024:.0f} KiB")
    if a.readme:
        block = to_ascii_stl(tris, "bla_di_like", compact=True)
        if check(block) != n:
            raise SystemExit("compact block does not round-trip")
        md = a.readme.read_text(encoding="utf-8")
        new, k = re.subn(r"```stl\n.*?```", lambda _: "```stl\n" + block + "```", md, flags=re.S)
        if k != 1:
            raise SystemExit(f"{a.readme}: expected exactly one ```stl block, found {k}")
        a.readme.write_text(new, encoding="utf-8", newline="\n")
        print(f"{a.readme}: ```stl block updated ({n} facets, one per line)")


if __name__ == "__main__":
    main()
