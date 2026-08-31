"""
Generate cross-platform icons from a single 1024x1024 PNG source.
Usage: python make_icons.py
Requires: pip install Pillow
Output:
  icons/logo.ico   - Windows (multi-size)
  icons/logo.icns  - macOS   (pure-Python ICNS, no macOS tools needed)
  icons/logo.png   - Linux   (256x256)
"""

import io
import struct
from pathlib import Path
from PIL import Image

SRC = Path("media/logo.png")

if not SRC.exists():
    raise FileNotFoundError(
        f"{SRC} not found.\n"
        "Export your logo as a 1024x1024 PNG with transparency and save it there."
    )

img = Image.open(SRC).convert("RGBA")
print(f"Source: {SRC} ({img.width}x{img.height})")


# ── Windows .ico
def make_ico(img: Image.Image, out: Path):
    sizes = [(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)]
    img.save(str(out), format="ICO", sizes=sizes)
    print(f"✓  {out}")

# ── macOS .icns (pure Python — no macOS tools required)
def make_icns(img: Image.Image, out: Path):
    # Type codes defined by Apple for PNG-format ICNS chunks
    SIZE_CODES = {
        16:   b'icp4',
        32:   b'icp5',
        64:   b'icp6',
        128:  b'ic07',
        256:  b'ic08',
        512:  b'ic09',
        1024: b'ic10',
    }
    chunks = bytearray()
    for size, code in SIZE_CODES.items():
        resized = img.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        data = buf.getvalue()
        chunk_len = 8 + len(data) # header (8) + data
        chunks += code + struct.pack(">I", chunk_len) + data

    total_len = 8 + len(chunks) # file header (8) + all chunks
    with open(out, "wb") as f:
        f.write(b"icns" + struct.pack(">I", total_len) + chunks)
    print(f"✓  {out}")

# ── Linux .png
def make_png(img: Image.Image, out: Path, size: int = 256):
    img.resize((size, size), Image.LANCZOS).save(str(out), format="PNG")
    print(f"✓  {out}")


make_ico(img,  Path("icons/logo.ico"))
make_icns(img, Path("icons/logo.icns"))
make_png(img,  Path("icons/logo.png"), 256)

print("\nDone. Commit icons/logo.ico, icons/logo.icns, and icons/logo.png.")
