# pip install Pillow numpy matplotlib

"""
perimeter_extraction.py

Walks a directory tree of sprite PNG files, traces the alpha-channel perimeter
of each frame to produce an ordered vertex contour, saves each contour as a
JSON file, and generates a verification plot for the first processed frame.

Contour extraction uses a direct pixel-boundary tracer rather than marching
squares.  Every vertex lies exactly on a pixel-grid corner (integer row/col
coordinate), so the resulting polygon is a blocky, stair-step silhouette that
matches the pixel art exactly.

Run from the perimeter_calculation/ directory:
    python perimeter_extraction.py
"""

import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Root directory containing character_* folders, relative to this script.
CHARACTERS_ROOT = "characters"

# Reference canvas dimensions (pixels).  These are kept for documentation only;
# normalization now uses each image's actual dimensions (see normalize_contour).
CANVAS_WIDTH = 32
CANVAS_HEIGHT = 32

# Alpha threshold: pixels with alpha strictly above this value are treated as
# opaque. Pixels at or below are transparent.
ALPHA_THRESHOLD = 0

# MULTI-CONTOUR EXTENSION POINT
# To support characters with disconnected silhouette parts (e.g. a held
# weapon that is visually separate from the body), set
# USE_LARGEST_CONTOUR_ONLY = False and update the JSON schema and the
# Blender assembly script to handle a list of contour lists rather than
# a single contour list.
USE_LARGEST_CONTOUR_ONLY = True

# ==============================================================================
# END CONFIGURATION
# ==============================================================================

# Cardinal direction vectors: (row_delta, col_delta)
_DIRS = {
    'E': (0,  +1),
    'S': (+1,  0),
    'W': (0,  -1),
    'N': (-1,  0),
}

# Right-hand turn priority for clockwise winding in image space (Y-down).
# Ensures the outer boundary is traced first when multiple outgoing edges meet
# at a corner (e.g. two pixel regions that touch only diagonally).
_CW_PRIORITY = {
    'E': ['S', 'E', 'N', 'W'],
    'S': ['W', 'S', 'E', 'N'],
    'W': ['N', 'W', 'S', 'E'],
    'N': ['E', 'N', 'W', 'S'],
}


def _get_dir(from_corner: tuple, to_corner: tuple) -> str:
    """Return the direction name ('N','S','E','W') for a unit step between corners.

    Args:
        from_corner: Starting (row, col) corner.
        to_corner:   Ending   (row, col) corner.

    Returns:
        Direction string.
    """
    dr = to_corner[0] - from_corner[0]
    dc = to_corner[1] - from_corner[1]
    for name, (r, c) in _DIRS.items():
        if dr == r and dc == c:
            return name
    raise ValueError(f"Not a unit step: {from_corner} -> {to_corner}")


def discover_frame_files(direction_dir: str, direction: str) -> list[tuple[int, str]]:
    """Return a sorted list of (frame_index, absolute_path) for all PNG frames
    in a direction folder.

    Naming conventions handled:
    - ``{direction}.png``      -> frame_index 0
    - ``{direction}{x}.png``   -> frame_index x  (x is a positive integer)

    Args:
        direction_dir: Absolute path to the direction folder.
        direction:     The direction name string (e.g. "south", "north_east").

    Returns:
        List of (frame_index, abs_path) tuples sorted by frame_index.
    """
    exact_name = f"{direction}.png"
    indexed_pattern = re.compile(rf"^{re.escape(direction)}(\d+)\.png$", re.IGNORECASE)

    frames: list[tuple[int, str]] = []

    for filename in os.listdir(direction_dir):
        abs_path = os.path.join(direction_dir, filename)
        if not os.path.isfile(abs_path):
            continue
        if filename == exact_name:
            frames.append((0, abs_path))
            continue
        m = indexed_pattern.match(filename)
        if m:
            frames.append((int(m.group(1)), abs_path))

    frames.sort(key=lambda t: t[0])
    return frames


def load_binary_mask(image_path: str) -> np.ndarray:
    """Load a PNG and return a binary alpha mask (1 = opaque, 0 = transparent).

    Args:
        image_path: Absolute path to the PNG file.

    Returns:
        A 2-D uint8 numpy array of shape (H, W).
    """
    img = Image.open(image_path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8)


def _collect_boundary_edges(mask: np.ndarray) -> dict:
    """Build a directed adjacency map of pixel-boundary half-edges.

    For each opaque pixel, emits one directed half-edge per exposed face.
    Winding convention: clockwise in image space (Y-down), meaning the opaque
    pixel is always to the RIGHT of the travel direction.

    Corner coordinate (r, c) is the top-left corner of pixel (r, c).
    Valid corner rows span 0..H and valid corner columns span 0..W.

    Args:
        mask: Binary 2-D uint8 array of shape (H, W).

    Returns:
        defaultdict mapping each corner tuple to a set of reachable corners.
    """
    H, W = mask.shape
    out_edges: dict[tuple, set] = defaultdict(set)

    for r in range(H):
        for c in range(W):
            if mask[r, c] == 0:
                continue
            # Top face exposed: travel East along the top edge
            if r == 0 or mask[r - 1, c] == 0:
                out_edges[(r, c)].add((r, c + 1))
            # Right face exposed: travel South along the right edge
            if c == W - 1 or mask[r, c + 1] == 0:
                out_edges[(r, c + 1)].add((r + 1, c + 1))
            # Bottom face exposed: travel West along the bottom edge
            if r == H - 1 or mask[r + 1, c] == 0:
                out_edges[(r + 1, c + 1)].add((r + 1, c))
            # Left face exposed: travel North along the left edge
            if c == 0 or mask[r, c - 1] == 0:
                out_edges[(r + 1, c)].add((r, c))

    return out_edges


def _trace_contours(out_edges: dict) -> list[list[tuple]]:
    """Trace all closed boundary loops from a directed edge map.

    Uses the right-hand rule (always prefer the most clockwise turn) so that
    the outer silhouette is captured as the first loop even when
    diagonal-touching pixel regions create ambiguous corners.

    Args:
        out_edges: defaultdict mapping corner to set of reachable corners.

    Returns:
        List of contours; each contour is an ordered list of (row, col) corners.
    """
    visited_edges: set[tuple] = set()
    contours: list[list[tuple]] = []

    for start in sorted(out_edges):
        for first_next in sorted(out_edges[start]):
            if (start, first_next) in visited_edges:
                continue

            path: list[tuple] = []
            current = start
            nxt = first_next
            inc_dir = _get_dir(current, nxt)

            while (current, nxt) not in visited_edges:
                visited_edges.add((current, nxt))
                path.append(current)
                current = nxt

                # Pick the next step using the right-hand (CW) priority
                candidates = out_edges.get(current, set())
                nxt = None
                for d in _CW_PRIORITY[inc_dir]:
                    dr, dc = _DIRS[d]
                    candidate = (current[0] + dr, current[1] + dc)
                    if candidate in candidates:
                        nxt = candidate
                        inc_dir = d
                        break

                if nxt is None:
                    break  # dead end — should not occur in a valid closed mask

            if len(path) >= 4:
                contours.append(path)

    return contours


def extract_pixel_boundary(mask: np.ndarray) -> list[tuple] | None:
    """Trace the pixel-exact boundary of a binary mask as a stair-step polygon.

    Every vertex lies exactly on a pixel-grid corner (integer row/col from 0
    to H or W), so the polygon follows pixel edges precisely — no smoothing or
    diagonal interpolation is applied.

    Args:
        mask: Binary 2-D uint8 array of shape (H, W).

    Returns:
        An ordered list of (row, col) corner tuples forming the contour, or
        None if no boundary edges were found.
    """
    out_edges = _collect_boundary_edges(mask)
    if not out_edges:
        return None

    contours = _trace_contours(out_edges)
    if not contours:
        return None

    # MULTI-CONTOUR EXTENSION POINT
    # To support characters with disconnected silhouette parts (e.g. a held
    # weapon that is visually separate from the body), set
    # USE_LARGEST_CONTOUR_ONLY = False and update the JSON schema and the
    # Blender assembly script to handle a list of contour lists rather than
    # a single contour list.
    if USE_LARGEST_CONTOUR_ONLY:
        return max(contours, key=len)
    else:
        return max(contours, key=len)  # placeholder — replace with full list handling


def normalize_contour(
    contour: list[tuple], width: int, height: int
) -> list[list[float]]:
    """Convert (row, col) pixel-grid corner coordinates to normalized [x, y] pairs.

    Corner (r, c) maps to x = c / width and y = r / height, placing all values
    in [0.0, 1.0].  Y is NOT flipped; the Blender assembly script handles the
    coordinate-system conversion.

    width and height must be the ACTUAL pixel dimensions of the source image so
    that sprites whose content reaches the image boundary (row == height or
    col == width) still produce coordinates in [0.0, 1.0].  Using a hardcoded
    constant smaller than the image causes y_norm > 1.0, which maps to out-of-
    bounds UV coordinates in the exported GLB and breaks texture rendering in
    Godot.

    Args:
        contour: List of (row, col) integer corner tuples.
        width:   Actual image width  in pixels (mask.shape[1]).
        height:  Actual image height in pixels (mask.shape[0]).

    Returns:
        List of [x_norm, y_norm] pairs, all values in [0.0, 1.0].
    """
    return [
        [round(c / float(width), 8), round(r / float(height), 8)]
        for r, c in contour
    ]


def build_output_path(
    characters_root: str,
    character_name: str,
    animation: str,
    direction: str,
    frame_index: int,
) -> str:
    """Compute the absolute path for the output JSON file.

    Args:
        characters_root: Absolute path to the characters/ root folder.
        character_name:  e.g. "character_1".
        animation:       e.g. "idle", "run".
        direction:       e.g. "south", "north_east".
        frame_index:     Integer frame index.

    Returns:
        Absolute path string for the JSON output file.
    """
    return os.path.join(
        characters_root,
        character_name,
        "mesh_definitions",
        f"{animation}_mesh",
        "directions",
        direction,
        f"{direction}{frame_index}.json",
    )


def build_verification_plot_path(
    characters_root: str,
    character_name: str,
    animation: str,
    direction: str,
    frame_index: int,
) -> str:
    """Compute the absolute path for the verification plot PNG.

    Args:
        characters_root: Absolute path to the characters/ root folder.
        character_name:  e.g. "character_1".
        animation:       e.g. "idle".
        direction:       e.g. "south".
        frame_index:     Integer frame index.

    Returns:
        Absolute path string for the verification plot file.
    """
    return os.path.join(
        characters_root,
        character_name,
        "mesh_definitions",
        f"{animation}_mesh",
        "directions",
        direction,
        f"verification_{direction}{frame_index}.png",
    )


def to_forward_slashes(path: str) -> str:
    """Replace backslashes with forward slashes for cross-platform JSON paths.

    Args:
        path: Any file path string.

    Returns:
        Path with all backslashes replaced by forward slashes.
    """
    return path.replace("\\", "/")


def write_json(
    output_path: str,
    character_name: str,
    animation: str,
    direction: str,
    frame_index: int,
    source_rel_path: str,
    normalized_contour: list[list[float]],
    canvas_width: int,
    canvas_height: int,
) -> None:
    """Write the contour data to a JSON file.

    Args:
        output_path:        Absolute path for the output file.
        character_name:     e.g. "character_1".
        animation:          e.g. "idle".
        direction:          e.g. "south".
        frame_index:        Integer frame index.
        source_rel_path:    Path to source PNG relative to perimeter_calculation/,
                            using forward slashes.
        normalized_contour: List of [x_norm, y_norm] pairs.
        canvas_width:       Actual image width  in pixels.
        canvas_height:      Actual image height in pixels.
    """
    payload = {
        "character": character_name,
        "animation": animation,
        "direction": direction,
        "frame_index": frame_index,
        "canvas_size": [canvas_width, canvas_height],
        "source_image": source_rel_path,
        "contour": normalized_contour,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def generate_verification_plot(
    image_path: str,
    normalized_contour: list[list[float]],
    plot_path: str,
    direction: str,
    frame_index: int,
) -> None:
    """Render and save a side-by-side verification plot for one frame.

    Left panel:  original RGBA source image.
    Right panel: normalized stair-step contour with vertex indices.

    Args:
        image_path:         Absolute path to the source PNG.
        normalized_contour: List of [x_norm, y_norm] pairs.
        plot_path:          Absolute path where the PNG plot will be saved.
        direction:          Direction name, used in panel labels.
        frame_index:        Frame index, used in panel labels.
    """
    img = Image.open(image_path).convert("RGBA")
    n_verts = len(normalized_contour)

    xs = [v[0] for v in normalized_contour]
    ys = [v[1] for v in normalized_contour]

    # Close the polygon
    xs_closed = xs + [xs[0]]
    ys_closed = ys + [ys[0]]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"{direction} | frame {frame_index}",
        fontsize=11,
        fontweight="bold",
    )

    # --- Left panel: source image ---
    ax_left.imshow(img)
    ax_left.set_title("Source Frame")
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # --- Right panel: stair-step contour ---
    ax_right.set_facecolor("white")
    ax_right.plot(xs_closed, ys_closed, color="steelblue", linewidth=1.0)
    ax_right.scatter(xs, ys, s=6, color="crimson", zorder=3)

    # Label every vertex for small contours; thin out for large ones to
    # avoid an unreadable cloud of overlapping numbers.
    label_step = max(1, n_verts // 40)
    for i in range(0, n_verts, label_step):
        x, y = xs[i], ys[i]
        ax_right.annotate(
            str(i),
            xy=(x, y),
            xytext=(x + 0.012, y - 0.012),
            fontsize=4,
            color="darkgreen",
        )

    ax_right.set_title(
        f"Extracted Contour (normalized, {n_verts} vertices)\n"
        "Y=0 is top-left (PNG convention, Y-flip not yet applied)",
        fontsize=8,
    )
    ax_right.set_xlim(0.0, 1.0)
    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_xlabel("X (normalized)")
    ax_right.set_ylabel("Y (normalized)")
    ax_right.invert_yaxis()  # match image-space: Y increases downward

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"VERIFICATION PLOT saved to: {to_forward_slashes(plot_path)}")
    print("Please review the plot and confirm the contour shape looks correct.")
    plt.show()
    plt.close(fig)


def process_frame(
    image_path: str,
    character_name: str,
    animation: str,
    direction: str,
    frame_index: int,
    script_dir: str,
    characters_root: str,
    first_plot_done: bool,
) -> tuple[bool, bool]:
    """Load, trace, normalize, and write JSON for a single sprite frame.

    Args:
        image_path:      Absolute path to the source PNG.
        character_name:  e.g. "character_1".
        animation:       e.g. "idle".
        direction:       e.g. "south".
        frame_index:     Integer frame index.
        script_dir:      Absolute path of the perimeter_calculation/ directory.
        characters_root: Absolute path to the characters/ root folder.
        first_plot_done: Whether the verification plot has already been saved.

    Returns:
        (processed, plot_done) booleans indicating if the frame was written
        successfully and if a verification plot was produced.
    """
    mask = load_binary_mask(image_path)
    img_h, img_w = mask.shape   # actual pixel dimensions of this specific image

    if mask.sum() == 0:
        print(f"WARNING: Fully transparent frame skipped: {to_forward_slashes(image_path)}")
        return False, first_plot_done

    if img_w != CANVAS_WIDTH or img_h != CANVAS_HEIGHT:
        print(
            f"WARNING: Canvas size mismatch — "
            f"{to_forward_slashes(image_path)} is {img_w}×{img_h} px "
            f"but the reference canvas is {CANVAS_WIDTH}×{CANVAS_HEIGHT} px. "
            "All animation frames should share the same canvas size for "
            "consistent character proportions in-game."
        )

    contour = extract_pixel_boundary(mask)
    if contour is None:
        print(f"WARNING: No contour found, skipped: {to_forward_slashes(image_path)}")
        return False, first_plot_done

    normalized = normalize_contour(contour, img_w, img_h)

    source_rel = to_forward_slashes(os.path.relpath(image_path, script_dir))

    output_path = build_output_path(
        characters_root, character_name, animation, direction, frame_index
    )
    write_json(
        output_path,
        character_name,
        animation,
        direction,
        frame_index,
        source_rel,
        normalized,
        img_w,
        img_h,
    )

    output_rel = to_forward_slashes(os.path.relpath(output_path, script_dir))
    print(f"Processed: {source_rel} -> {output_rel} ({len(normalized)} vertices)")

    plot_produced = False
    if not first_plot_done:
        plot_path = build_verification_plot_path(
            characters_root, character_name, animation, direction, frame_index
        )
        generate_verification_plot(
            image_path, normalized, plot_path, direction, frame_index
        )
        plot_produced = True

    return True, first_plot_done or plot_produced


def main() -> None:
    """Entry point: walk all character/animation/direction trees and process frames."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    characters_root = os.path.join(script_dir, CHARACTERS_ROOT)

    if not os.path.isdir(characters_root):
        raise FileNotFoundError(f"Characters root not found: {characters_root}")

    char_dirs = sorted(
        entry.name
        for entry in os.scandir(characters_root)
        if entry.is_dir() and re.match(r"^character_\d+$", entry.name)
    )

    total_processed = 0
    total_skipped = 0
    first_plot_done = False

    for character_name in char_dirs:
        char_path = os.path.join(characters_root, character_name)

        anim_dirs = sorted(
            entry.name
            for entry in os.scandir(char_path)
            if entry.is_dir() and entry.name.endswith("_frames")
        )

        for anim_folder in anim_dirs:
            animation = anim_folder[: -len("_frames")]
            directions_path = os.path.join(char_path, anim_folder, "directions")

            if not os.path.isdir(directions_path):
                continue

            direction_names = sorted(
                entry.name
                for entry in os.scandir(directions_path)
                if entry.is_dir()
            )

            for direction in direction_names:
                direction_dir = os.path.join(directions_path, direction)
                frames = discover_frame_files(direction_dir, direction)

                for frame_index, image_path in frames:
                    processed, first_plot_done = process_frame(
                        image_path=image_path,
                        character_name=character_name,
                        animation=animation,
                        direction=direction,
                        frame_index=frame_index,
                        script_dir=script_dir,
                        characters_root=characters_root,
                        first_plot_done=first_plot_done,
                    )
                    if processed:
                        total_processed += 1
                    else:
                        total_skipped += 1

    print(f"\nRun complete. {total_processed} frames processed, {total_skipped} skipped.")


if __name__ == "__main__":
    main()
