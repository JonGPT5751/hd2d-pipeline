"""
fix_godot_texture_imports.py

Patches every *.png.import file under the Godot glb_files asset directory so
that textures are imported as Lossless (compress/mode=0) instead of VRAM-
Compressed (S3TC / compress/mode=2).

Why this is needed
------------------
Godot regenerates .import files from scratch whenever a new asset folder is
imported (e.g. after running blender_mesh_builder.py and copying the GLBs into
the project).  The freshly generated files use Godot's built-in default, which
is VRAM Compressed (S3TC block compression).  S3TC works in 4×4 pixel blocks,
so it completely destroys 32×32 pixel art — colours shift and patterns break.

Changing Project Settings → Import Defaults does NOT fix already-imported
files; it only affects files with no .import file yet.  This script edits the
files directly so Godot recompiles them as lossless on the next editor load.

Usage
-----
Run once after every GLB reimport into Godot:

    python fix_godot_texture_imports.py

Then switch back to the Godot editor — it will detect the changed .import files
and recompile the textures automatically.
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — adjust GLB_ROOT if your Godot project lives elsewhere.
# ---------------------------------------------------------------------------
GLB_ROOT = Path(r"C:\Users\Jowhi\OneDrive\Documents\3d-hearthbound\assets\glb_files")
# ---------------------------------------------------------------------------


def patch_import_files(glb_root: Path) -> None:
    if not glb_root.exists():
        print(f"ERROR: GLB root not found: {glb_root}")
        sys.exit(1)

    imports = list(glb_root.rglob("*.png.import"))
    if not imports:
        print(f"No .png.import files found under {glb_root}")
        return

    print(f"Found {len(imports)} .import file(s) under {glb_root}")

    fixed = 0
    for p in imports:
        text = p.read_text(encoding="utf-8")
        original = text

        # Lossless compression — no block artefacts on pixel art
        text = re.sub(r"compress/mode=\d+", "compress/mode=0", text)
        # Mipmaps blur pixel art at a distance; disable them
        text = re.sub(r"mipmaps/generate=\w+", "mipmaps/generate=false", text)
        # high_quality flag is irrelevant for lossless but keep it tidy
        text = re.sub(r"compress/high_quality=\w+", "compress/high_quality=false", text)

        if text != original:
            p.write_text(text, encoding="utf-8")
            fixed += 1

    print(f"Patched {fixed} file(s)  ({len(imports) - fixed} already correct).")

    if imports:
        print("\nSpot-check on first file:")
        sample = imports[0].read_text(encoding="utf-8")
        for line in sample.splitlines():
            if any(k in line for k in ("compress/mode", "mipmaps/generate", "vram_texture")):
                print(f"  {line}")

    print("\nDone.  Switch to the Godot editor — it will recompile the textures automatically.")


if __name__ == "__main__":
    patch_import_files(GLB_ROOT)
