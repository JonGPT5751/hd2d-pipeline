# Run with: blender --background --python blender_mesh_builder.py
# Must be run from the perimeter_calculation/ directory

"""
blender_mesh_builder.py

Runs perimeter_extraction.py as a subprocess to ensure all JSON mesh definition
files are up to date, then reads every JSON file found under the
mesh_definitions/ directories and builds a corresponding Blender mesh for each
one, saving each mesh as its own .blend file and exporting it as a binary .glb.

Each mesh is a thin extruded polygon whose front face carries the sprite PNG
texture (alpha-clipped) and whose remaining faces are flat black.  Before the
.glb export the object origin is standardised to the bottom-centre of the mesh
bounding box so the character's feet sit at the local origin.

Tested against Blender 5.1.1.
"""

import bpy
import bmesh
import math
import os
import json
import sys
import subprocess
import shutil
import re
import mathutils
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to perimeter_extraction.py (relative to this script)
PERIMETER_SCRIPT = "perimeter_extraction.py"

# Root characters directory (relative to this script)
CHARACTERS_DIR = "characters"

# Extrusion thickness divisor (thickness = 1.0 / THICKNESS_DIVISOR)
THICKNESS_DIVISOR = 128

# Root tilesets directory (relative to this script)
TILESETS_DIR = "tilesets"

# ==============================================================================
# END CONFIGURATION
# ==============================================================================

# Ordered face names used as material slots for tile cubes (slot index = list index)
_TILE_FACE_NAMES = ("top", "bottom", "front", "back", "right", "left")


def run_perimeter_extraction(script_dir: Path) -> None:
    """Run perimeter_extraction.py as a subprocess to refresh all JSON files.

    Blender's bundled Python (sys.executable) usually lacks Pillow, numpy, and
    matplotlib.  To avoid forcing a manual pre-step, this function builds a
    list of candidate interpreters and tries each one in order until one
    succeeds:

        1. sys.executable  — Blender's Python (works if packages are installed
                             in Blender's environment, rare but possible).
        2. "python"        — whatever ``python`` resolves to on the system PATH.
        3. "python3"       — common on macOS / Linux.
        4. "py"            — Windows Python Launcher shorthand.

    Duplicates (e.g. if "python" resolves to the same path as sys.executable)
    are skipped automatically.

    Args:
        script_dir: Absolute path to the directory containing both scripts.

    Raises:
        SystemExit: If every interpreter fails AND no JSON files exist yet.
    """
    script_path = script_dir / PERIMETER_SCRIPT
    print(f"Running perimeter extraction: {script_path.as_posix()}")

    # Build a de-duplicated list of candidate Python interpreters.
    seen: set = set()
    candidates: list = []
    for name in (sys.executable, "python", "python3", "py"):
        resolved = shutil.which(name) or name
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    returncode = 1
    for interpreter in candidates:
        print(f"  Trying interpreter: {interpreter}")
        try:
            result = subprocess.run(
                [interpreter, str(script_path)],
                cwd=str(script_dir),
            )
            returncode = result.returncode
        except OSError as exc:
            print(f"  Could not launch '{interpreter}': {exc}")
            returncode = 1

        if returncode == 0:
            return  # success — JSON files are up to date

        print(f"  '{interpreter}' exited with code {returncode}, trying next…")

    # Every interpreter failed — fall back to existing JSON files if present.
    existing = list((script_dir / CHARACTERS_DIR).rglob(
        "mesh_definitions/**/*.json"
    ))
    if existing:
        print(
            f"WARNING: {PERIMETER_SCRIPT} failed on all available interpreters "
            f"but {len(existing)} existing JSON file(s) found — "
            "continuing mesh build with current data.  "
            "Install Pillow, numpy, and matplotlib into your system Python "
            "('python -m pip install Pillow numpy matplotlib') to enable "
            "automatic regeneration."
        )
    else:
        print(
            f"ERROR: {PERIMETER_SCRIPT} failed on all available interpreters "
            "and no JSON files exist.  Install Pillow, numpy, and matplotlib "
            "into your system Python and run perimeter_extraction.py manually "
            "to generate the initial JSON files."
        )
        sys.exit(1)


def discover_json_files(characters_root: Path) -> list:
    """Walk characters_root and return all .json files under mesh_definitions/.

    Only directories matching the pattern character_{n} are entered.

    Args:
        characters_root: Absolute path to the characters/ root directory.

    Returns:
        Sorted list of absolute Path objects for every JSON file found.
    """
    json_files = []
    if not characters_root.is_dir():
        return json_files

    for char_dir in sorted(characters_root.iterdir()):
        if not char_dir.is_dir():
            continue
        if not re.match(r"^character_\d+$", char_dir.name):
            continue
        mesh_defs = char_dir / "mesh_definitions"
        if not mesh_defs.is_dir():
            continue
        for json_path in sorted(mesh_defs.rglob("*.json")):
            json_files.append(json_path)

    return json_files


def json_to_glb_path(json_path: Path, script_dir: Path) -> Path:
    """Derive the output .glb path that mirrors the input JSON path.

    Replaces the ``mesh_definitions`` path component with ``glb_files`` and
    changes the file extension from ``.json`` to ``.glb``.

    Args:
        json_path:  Absolute path to the source JSON file.
        script_dir: Absolute path to the script's working directory.

    Returns:
        Absolute Path for the output .glb file.
    """
    rel = json_path.relative_to(script_dir)
    new_parts = tuple(
        "glb_files" if part == "mesh_definitions" else part
        for part in rel.parts
    )
    glb_rel = Path(*new_parts).with_suffix(".glb")
    return script_dir / glb_rel


def json_to_blend_path(json_path: Path, script_dir: Path) -> Path:
    """Derive the output .blend path that mirrors the input JSON path.

    Replaces the ``mesh_definitions`` path component with ``blend_files`` and
    changes the file extension from ``.json`` to ``.blend``.

    Args:
        json_path:  Absolute path to the source JSON file.
        script_dir: Absolute path to the script's working directory.

    Returns:
        Absolute Path for the output .blend file.
    """
    rel = json_path.relative_to(script_dir)
    new_parts = tuple(
        "blend_files" if part == "mesh_definitions" else part
        for part in rel.parts
    )
    blend_rel = Path(*new_parts).with_suffix(".blend")
    return script_dir / blend_rel


def clear_scene() -> None:
    """Remove all objects, meshes, materials, and images from the current scene.

    Called before processing each JSON file to prevent data bleed between
    successive mesh builds.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)


def set_origin_to_bottom_center(obj: bpy.types.Object) -> None:
    """Move the object's origin to the bottom-centre of its bounding box.

    Translates the mesh vertices in local space so the bottom-centre of the
    bounding box sits at the local origin, then updates obj.location to
    compensate so all world-space positions remain unchanged.

    "Bottom-centre" is defined as:
        x = centre of X extent  (horizontal midpoint of the silhouette)
        y = minimum Y           (feet of the character in Blender Y-up space)
        z = centre of Z extent  (midpoint through the extrusion depth)

    Args:
        obj: The Blender mesh object to modify in place.
    """
    corners = [mathutils.Vector(c) for c in obj.bound_box]
    x_min = min(v.x for v in corners)
    x_max = max(v.x for v in corners)
    y_min = min(v.y for v in corners)
    z_min = min(v.z for v in corners)
    z_max = max(v.z for v in corners)

    offset = mathutils.Vector((
        (x_min + x_max) / 2.0,
        y_min,
        (z_min + z_max) / 2.0,
    ))

    # Shift all vertices so bottom-centre lands at the local origin
    obj.data.transform(mathutils.Matrix.Translation(-offset))
    obj.data.update()

    # Move the object to keep its world-space geometry in place
    obj.location = obj.location + offset


def export_glb(obj: bpy.types.Object, glb_path: Path) -> None:
    """Export one object as a self-contained binary .glb file.

    Textures are embedded directly in the binary payload (not saved as
    external files).  Only the supplied object is exported; all modifiers
    are applied.

    Args:
        obj:      The Blender object to export.
        glb_path: Absolute path for the output .glb file.
    """
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format='GLB',          # Binary: all data in one file
        use_selection=True,           # Only the active/selected object
        export_apply=True,            # Apply modifiers before export
        export_texcoords=True,        # Include UV maps
        export_normals=True,          # Include normals
        export_materials='EXPORT',    # Include materials
        export_image_format='AUTO',   # PNG for RGBA images (preserves alpha)
        export_keep_originals=False,  # Embed textures, no external files
        export_yup=True,              # Y-up coordinate system (glTF standard)
    )


def create_sprite_front_material(image: bpy.types.Image) -> bpy.types.Material:
    """Create a Principled BSDF material that displays the sprite PNG texture.

    The image's Alpha output drives the shader's Alpha input so that
    transparent pixels are cut out.  blend_method is set to CLIP and
    use_transparent_shadow is enabled so the silhouette casts correct shadows.

    Args:
        image: Blender Image datablock to use as the colour + alpha texture.

    Returns:
        A fully configured bpy.types.Material named ``sprite_front``.
    """
    mat = bpy.data.materials.new(name="sprite_front")
    mat.use_nodes = True

    nt = mat.node_tree
    nt.nodes.clear()

    # Principled BSDF
    principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Specular IOR Level"].default_value = 0.0

    # Image Texture — Closest interpolation keeps pixel art sharp (no bilinear blur)
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = image
    tex_node.interpolation = "Closest"

    # Material Output
    output_node = nt.nodes.new("ShaderNodeOutputMaterial")

    # Wire colour and alpha through to the shader
    nt.links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])
    nt.links.new(tex_node.outputs["Alpha"], principled.inputs["Alpha"])
    nt.links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])

    # Transparency settings (Blender 5.x: shadow_method removed)
    mat.blend_method = "CLIP"
    mat.use_transparent_shadow = True

    return mat


def create_black_back_material() -> bpy.types.Material:
    """Create a flat pure-black Principled BSDF material for back and side faces.

    No texture, no emission.  All non-listed inputs remain at their defaults.

    Returns:
        A fully configured bpy.types.Material named ``black_back``.
    """
    mat = bpy.data.materials.new(name="black_back")
    mat.use_nodes = True

    nt = mat.node_tree
    nt.nodes.clear()

    # Principled BSDF
    principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Specular IOR Level"].default_value = 0.0

    # Material Output
    output_node = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])

    return mat


def build_mesh_from_json(json_path: Path, script_dir: Path) -> bpy.types.Object:
    """Build a Blender mesh object from one JSON mesh definition file.

    Steps performed:
    1. Load and validate the JSON contour.
    2. Create front-face vertices with Y-flip (PNG -> Blender coordinate space).
    3. Build the front face as a single n-gon.
    4. Extrude the face by ``thickness`` along -Z.
    5. Call recalc_face_normals to fix outward orientation (extrude_face_region
       inverts the original face's winding, so normals must be recalculated).
    6. Classify faces: front (Z~0, normal +Z) vs. back/sides; assign UVs to
       the front face (u = x, v = y of each vertex).
    7. Create and assign sprite_front (slot 0) and black_back (slot 1) materials.
    8. Link the object to the active scene collection.

    Args:
        json_path:  Absolute path to the JSON mesh definition file.
        script_dir: Absolute path to the script's working directory.

    Returns:
        The created bpy.types.Object, already linked to the scene collection.

    Raises:
        ValueError:       If the contour has fewer than 3 vertices.
        FileNotFoundError: If the referenced source image does not exist on disk.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    character   = data["character"]
    animation   = data["animation"]
    direction   = data["direction"]
    frame_index = data["frame_index"]
    source_rel  = data["source_image"]
    contour     = data["contour"]

    # --- Validation ---------------------------------------------------
    if len(contour) < 3:
        raise ValueError(
            f"Contour has only {len(contour)} vertices; at least 3 required"
        )

    source_abs = (script_dir / source_rel).resolve()
    if not source_abs.exists():
        raise FileNotFoundError(
            f"Source image not found: {source_abs.as_posix()}"
        )

    object_name = f"{character}_{animation}_{direction}_{frame_index}"

    # --- Build geometry with bmesh ------------------------------------
    bm = bmesh.new()

    # Front-face vertices: apply Y-flip (PNG Y-down -> Blender Y-up)
    verts = []
    for x_norm, y_norm in contour:
        verts.append(bm.verts.new((x_norm, 1.0 - y_norm, 0.0)))
    bm.verts.ensure_lookup_table()

    # Single n-gon front face
    front_face = bm.faces.new(verts)
    bm.faces.ensure_lookup_table()

    # Extrude the front face backward along -Z.
    # Note: extrude_face_region inverts the original face's winding, so both
    # cap normals point inward immediately after the translate.  We fix them
    # below with recalc_face_normals before any normal-dependent work.
    thickness = 1.0 / THICKNESS_DIVISOR
    extrude_result = bmesh.ops.extrude_face_region(bm, geom=[front_face])
    extruded_verts = [
        g for g in extrude_result["geom"]
        if isinstance(g, bmesh.types.BMVert)
    ]
    bmesh.ops.translate(bm, verts=extruded_verts, vec=(0.0, 0.0, -thickness))

    # Recalculate all normals to point outward from the closed solid.
    # After this: z=0 cap → normal +Z, z=-thickness cap → normal -Z.
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    # --- UV map + material index assignment ---------------------------
    # Front face: Z centre ~ 0.0 AND normal Z > 0.9  -> slot 0 (sprite_front)
    # All other faces                                 -> slot 1 (black_back)
    # UVs are set only on the front face; back/side faces need no texture.
    uv_layer = bm.loops.layers.uv.new("UVMap")
    bm.faces.ensure_lookup_table()
    z_eps = thickness * 0.1
    for face in bm.faces:
        center_z = face.calc_center_median().z
        face.normal_update()
        if abs(center_z) < z_eps and face.normal.z > 0.9:
            face.material_index = 0
            for loop in face.loops:
                loop[uv_layer].uv = (loop.vert.co.x, loop.vert.co.y)
        else:
            face.material_index = 1

    # --- Write bmesh to a real mesh datablock -------------------------
    mesh_data = bpy.data.meshes.new(object_name)
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()

    # --- Create and link scene object ---------------------------------
    obj = bpy.data.objects.new(object_name, mesh_data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # --- Materials ----------------------------------------------------
    image = bpy.data.images.load(str(source_abs))
    image.colorspace_settings.name = 'sRGB'

    mat_front = create_sprite_front_material(image)
    mat_back  = create_black_back_material()

    obj.data.materials.append(mat_front)  # slot 0
    obj.data.materials.append(mat_back)   # slot 1

    return obj


# ==============================================================================
# TILESET PIPELINE — unit-cube mesh builder
# ==============================================================================


def _classify_face(normal: mathutils.Vector) -> str:
    """Return the face name for a cube face based on its outward normal.

    Uses a dot-product threshold of 0.9 so that axis-aligned normals are
    classified unambiguously.  The caller should ensure normals are freshly
    computed (face.normal_update()) before passing them here.

    Args:
        normal: The face normal in local object space.

    Returns:
        One of "top", "bottom", "front", "back", "right", "left", or
        "unknown" if no axis is dominant.
    """
    if normal.y > 0.9:   return "top"
    if normal.y < -0.9:  return "bottom"
    if normal.z > 0.9:   return "front"
    if normal.z < -0.9:  return "back"
    if normal.x > 0.9:   return "right"
    if normal.x < -0.9:  return "left"
    return "unknown"


def _compute_face_uv(face_name: str, co: mathutils.Vector) -> tuple:
    """Compute box-projection UV coordinates for a vertex on a named cube face.

    All cube vertices are at ±0.5, so each formula maps the two non-dominant
    axes onto the full [0, 1] UV range.  The V values are chosen so that,
    after the glTF exporter's automatic V-flip (v_glTF = 1 − v_Blender), the
    top of the face texture aligns with the top of the face in Godot.

    Face formulae (Blender local coords, before any rotation):
        top    (+Y normal):  U = x + 0.5,   V = −z + 0.5
        bottom (−Y normal):  U = x + 0.5,   V =  z + 0.5
        front  (+Z normal):  U = x + 0.5,   V =  y + 0.5
        back   (−Z normal):  U = −x + 0.5,  V =  y + 0.5
        right  (+X normal):  U = −z + 0.5,  V =  y + 0.5
        left   (−X normal):  U =  z + 0.5,  V =  y + 0.5

    Args:
        face_name: One of the six face name strings.
        co:        Vertex position in local object space.

    Returns:
        (u, v) tuple in [0, 1] × [0, 1].
    """
    x, y, z = co.x, co.y, co.z
    if face_name == "top":     return (x + 0.5,  -z + 0.5)
    if face_name == "bottom":  return (x + 0.5,   z + 0.5)
    if face_name == "front":   return (x + 0.5,   y + 0.5)
    if face_name == "back":    return (-x + 0.5,  y + 0.5)
    if face_name == "right":   return (-z + 0.5,  y + 0.5)
    if face_name == "left":    return (z + 0.5,   y + 0.5)
    return (0.5, 0.5)  # fallback for "unknown"


def discover_tileset_dirs(tilesets_root: Path) -> list:
    """Return sorted list of tileset directories matching ``tileset_{n}``.

    Only immediate children of *tilesets_root* that are directories and
    whose names match the pattern are returned (no recursive search).

    Args:
        tilesets_root: Absolute path to the tilesets/ root directory.

    Returns:
        Sorted list of absolute Path objects, one per discovered tileset.
    """
    if not tilesets_root.is_dir():
        return []
    result = []
    for entry in sorted(tilesets_root.iterdir()):
        if entry.is_dir() and re.match(r"^tileset_\d+$", entry.name):
            result.append(entry)
    return result


def create_tile_material(
    face_name: str, image: bpy.types.Image
) -> bpy.types.Material:
    """Create a Principled BSDF material textured with the given sRGB image.

    Uses Closest interpolation to keep pixel-art tiles sharp.  The material
    is fully opaque (OPAQUE blend mode); tiles have no alpha cut-out.

    Args:
        face_name: Short label used as part of the material name.
        image:     Blender Image datablock, already loaded and set to sRGB.

    Returns:
        A configured bpy.types.Material named ``tile_{face_name}``.
    """
    mat = bpy.data.materials.new(name=f"tile_{face_name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Specular IOR Level"].default_value = 0.0

    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = image
    tex_node.interpolation = "Closest"

    output_node = nt.nodes.new("ShaderNodeOutputMaterial")

    nt.links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])
    nt.links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])

    mat.blend_method = "OPAQUE"
    return mat


def create_tile_fallback_material(face_name: str) -> bpy.types.Material:
    """Create a mid-gray Principled BSDF material for a missing tile texture.

    Used whenever the expected ``{face_name}.png`` file does not exist in
    the tileset directory, so the export can still proceed with a visible
    placeholder.

    Args:
        face_name: Short label used as part of the material name.

    Returns:
        A configured bpy.types.Material named ``tile_{face_name}_fallback``.
    """
    mat = bpy.data.materials.new(name=f"tile_{face_name}_fallback")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Specular IOR Level"].default_value = 0.0

    output_node = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])

    mat.blend_method = "OPAQUE"
    return mat


def export_tile_glb(obj: bpy.types.Object, glb_path: Path) -> None:
    """Export one tile object as a self-contained binary .glb file.

    Prefers JPEG image encoding to keep file sizes small; falls back to
    AUTO (PNG for RGBA) if the installed Blender version does not accept
    ``export_image_format='JPEG'``.

    Args:
        obj:      The Blender tile object to export.
        glb_path: Absolute path for the output .glb file.
    """
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    common_kwargs = dict(
        filepath=str(glb_path),
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_keep_originals=False,
        export_yup=True,
    )
    try:
        bpy.ops.export_scene.gltf(**common_kwargs, export_image_format='JPEG')
    except TypeError:
        bpy.ops.export_scene.gltf(**common_kwargs, export_image_format='AUTO')


def build_tile_mesh(
    tileset_dir: Path, script_dir: Path
) -> bpy.types.Object:
    """Build a textured 1×1×1 unit cube for the given tileset directory.

    Construction steps:
    1.  Create a unit cube with ``bmesh.ops.create_cube(size=1.0)``.
        Vertices start at ±0.5 on every axis.
    2.  Classify each face by its outward normal (+Y → top, +Z → front, etc.).
    3.  Assign per-face box-projection UV coordinates and material slot indices.
    4.  Write the bmesh to a mesh datablock.
    5.  Translate the mesh geometry +0.5 in Y so the origin sits at the
        centre of the bottom face (tile bottom at Y = 0).
    6.  Link the object to the active scene collection.
    7.  Append one material per face slot (ordered: top, bottom, front, back,
        right, left).  Missing ``{face}.png`` files receive a gray fallback.

    A 90° X-axis rotation is applied and baked in ``process_tilesets`` after
    this function returns, mapping Blender +Y (up) to Blender world +Z so
    that ``export_yup=True`` correctly places the top face as +Y in Godot.

    Args:
        tileset_dir: Path to the ``tileset_{n}`` directory.
        script_dir:  Path to the script's working directory (unused here but
                     kept for API symmetry with ``build_mesh_from_json``).

    Returns:
        The created bpy.types.Object, already linked to the scene collection.
    """
    tileset_name = tileset_dir.name  # e.g. "tileset_1"

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Map each face to its name and material slot index
    face_slot = {name: idx for idx, name in enumerate(_TILE_FACE_NAMES)}
    face_names_by_index: dict = {}

    for i, face in enumerate(bm.faces):
        face.normal_update()
        name = _classify_face(face.normal)
        face_names_by_index[i] = name
        face.material_index = face_slot.get(name, 0)

    # Assign per-face box-projection UV coordinates
    uv_layer = bm.loops.layers.uv.new("UVMap")
    bm.faces.ensure_lookup_table()
    for i, face in enumerate(bm.faces):
        fname = face_names_by_index[i]
        for loop in face.loops:
            loop[uv_layer].uv = _compute_face_uv(fname, loop.vert.co)

    # Write bmesh to a real mesh datablock
    mesh_data = bpy.data.meshes.new(tileset_name)
    bm.to_mesh(mesh_data)
    bm.free()

    # Translate geometry +0.5 in Y → origin sits at the centre of the bottom face
    mesh_data.transform(
        mathutils.Matrix.Translation(mathutils.Vector((0.0, 0.5, 0.0)))
    )
    mesh_data.update()

    # Create and link scene object
    obj = bpy.data.objects.new(tileset_name, mesh_data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Append one material per face slot in canonical order
    for face_name in _TILE_FACE_NAMES:
        tex_path = tileset_dir / f"{face_name}.png"
        if tex_path.exists():
            image = bpy.data.images.load(str(tex_path))
            image.colorspace_settings.name = 'sRGB'
            mat = create_tile_material(face_name, image)
        else:
            print(
                f"  WARNING: {tex_path.as_posix()} not found — "
                f"using gray fallback for '{face_name}' face."
            )
            mat = create_tile_fallback_material(face_name)
        obj.data.materials.append(mat)

    return obj


def process_tilesets(script_dir: Path) -> tuple:
    """Discover every ``tileset_{n}`` directory and build its tile mesh.

    For each discovered tileset:
    - Clears the Blender scene.
    - Builds a textured unit cube with ``build_tile_mesh``.
    - Applies a 90° X-axis rotation so that the +Y (Blender up) face
      becomes +Z in Blender world space, which ``export_yup=True`` then
      maps to +Y (up) in Godot — making the tile sit flat on the ground.
    - Saves the result as a ``.blend`` file.
    - Exports it as a ``.glb`` file.

    Output paths per tileset:
        .blend → ``tilesets/tileset_{n}/blend_files/tile_mesh/tileset_{n}.blend``
        .glb   → ``tilesets/tileset_{n}/glb_files/tile_mesh/tileset_{n}.glb``

    Args:
        script_dir: Absolute path to the script's working directory.

    Returns:
        ``(total_built, total_failed)`` counts as a tuple of ints.
    """
    tilesets_root = script_dir / TILESETS_DIR
    tileset_dirs = discover_tileset_dirs(tilesets_root)

    if not tileset_dirs:
        print(f"No tileset directories found under {tilesets_root.as_posix()}")
        return (0, 0)

    print(f"\nFound {len(tileset_dirs)} tileset(s) under {tilesets_root.as_posix()}")

    total_built  = 0
    total_failed = 0

    for tileset_dir in tileset_dirs:
        tileset_name = tileset_dir.name
        blend_path = (
            tileset_dir / "blend_files" / "tile_mesh" / f"{tileset_name}.blend"
        )
        glb_path = (
            tileset_dir / "glb_files" / "tile_mesh" / f"{tileset_name}.glb"
        )
        tileset_rel = tileset_dir.relative_to(script_dir).as_posix()
        blend_rel   = blend_path.relative_to(script_dir).as_posix()
        glb_rel     = glb_path.relative_to(script_dir).as_posix()

        try:
            clear_scene()
            obj = build_tile_mesh(tileset_dir, script_dir)

            # Rotate 90° on X: maps Blender +Y (cube "up") to world +Z,
            # which export_yup then maps to Godot +Y (up) — tile sits flat.
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            obj.rotation_euler.x = math.radians(90)
            bpy.ops.object.transform_apply(rotation=True)

            blend_path.parent.mkdir(parents=True, exist_ok=True)
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)
            print(f"Built:    {tileset_rel} -> {blend_rel}")

            export_tile_glb(obj, glb_path)
            print(f"Exported: {tileset_rel} -> {glb_rel}")

            total_built += 1

        except Exception as exc:
            print(f"ERROR:    {tileset_rel} -- {exc}")
            total_failed += 1

    return (total_built, total_failed)


def main() -> None:
    """Entry point: run extraction, build all meshes, save .blend files."""
    try:
        raw = Path(os.path.abspath(__file__))
        # When the script is run from Blender's Scripting workspace the text
        # block is stored inside the active .blend file, producing a __file__
        # like:  …/some_name.blend/blender_mesh_builder.py
        # Detect this by checking for a .blend segment in the path and fall
        # back to the process working directory instead.
        if any(part.endswith(".blend") for part in raw.parts):
            raise ValueError("script is embedded inside a .blend file")
        script_dir = raw.parent
    except (NameError, ValueError):
        script_dir = Path(os.getcwd())

    # ------------------------------------------------------------------
    # Step 1 — Ensure JSON mesh definitions are up to date
    # ------------------------------------------------------------------
    run_perimeter_extraction(script_dir)

    # ------------------------------------------------------------------
    # Step 2 — Discover all JSON files
    # ------------------------------------------------------------------
    characters_root = script_dir / CHARACTERS_DIR
    json_files = discover_json_files(characters_root)

    if not json_files:
        print("No JSON mesh definition files found — skipping character mesh step.")

    total_built       = 0
    total_failed      = 0
    total_glb         = 0
    total_glb_failed  = 0

    for json_path in json_files:
        blend_path = json_to_blend_path(json_path, script_dir)
        glb_path   = json_to_glb_path(json_path, script_dir)
        json_rel   = json_path.relative_to(script_dir).as_posix()
        blend_rel  = blend_path.relative_to(script_dir).as_posix()
        glb_rel    = glb_path.relative_to(script_dir).as_posix()

        # --- Build mesh and save .blend -----------------------------------
        try:
            clear_scene()
            obj = build_mesh_from_json(json_path, script_dir)

            blend_path.parent.mkdir(parents=True, exist_ok=True)
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)

            print(f"Built:    {json_rel} -> {blend_rel}")
            total_built += 1

        except Exception as exc:
            print(f"ERROR:    {json_rel} -- {exc}")
            total_failed += 1
            continue  # skip GLB if the mesh build itself failed

        # --- Rotate, bake, set origin, export .glb ------------------------
        try:
            # Rotate 90° on X so the character stands upright in Godot.
            # The mesh is built in the Blender XY plane (Y = up in sprite
            # space); this rotation maps Y → Z (world up after Y-up export).
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            obj.rotation_euler.x = math.radians(90)
            bpy.ops.object.transform_apply(rotation=True)

            set_origin_to_bottom_center(obj)
            export_glb(obj, glb_path)

            print(f"Exported: {json_rel} -> {glb_rel}")
            total_glb += 1

        except Exception as exc:
            print(f"ERROR (glb): {glb_rel} -- {exc}")
            total_glb_failed += 1

    # ------------------------------------------------------------------
    # Step 3 — Build tile meshes from tilesets/
    # ------------------------------------------------------------------
    tile_built, tile_failed = process_tilesets(script_dir)

    print(
        f"\nBuild complete. "
        f"{total_built} character mesh(es) built, {total_failed} failed. "
        f"{total_glb} character GLB(s) exported, {total_glb_failed} failed. "
        f"{tile_built} tile mesh(es) built, {tile_failed} failed."
    )


if __name__ == "__main__":
    main()
