import bpy
import gpu
import colorsys
import math
from mathutils import Vector
from gpu_extras.batch import batch_for_shader

from .origami_state import DEBUG_UV_SEGMENTS
from .origami_state import INTERACTION_UV_SEGMENT_LISTS
from .origami_state import DEBUG_BLOCKED_EDGES
from .origami_state import DEBUG_FLOODS
from .origami_state import DEBUG_TESTED_EDGES
from .origami_state import DEBUG_CORNER_POINTS

# Store handler globally so we can remove it
_crease_handle = None
_uv_debug_handle = None

# -----------------------------
# Data Extraction
# -----------------------------

def get_crease_lines(obj):
    """Extract crease lines from fold history."""
    lines = []

    if not hasattr(obj, "fold_history"):
        return lines

    for f in obj.fold_history:
        pivot = Vector(f.pivot_uv)
        direction = Vector(f.dir_uv)

        if direction.length == 0:
            continue

        direction.normalize()

        lines.append({
            "pivot": pivot,
            "dir": direction,
            "angle": f.angle,
        })

    return lines


# -----------------------------
# Geometry: Clip to UV bounds
# -----------------------------

def debug_corner_fold_uv(obj, bm, corner):
    from .origami_state import DEBUG_CORNER_POINTS

    DEBUG_CORNER_POINTS.clear()

    uv_layer = bm.loops.layers.uv.active

    regions = [
        (corner.region_a_faces, (1, 1, 0, 1)),
        (corner.region_b_faces, (0, 1, 1, 1)),
        (corner.overlap_faces_a, (0, 1, 0, 1)),
        (corner.overlap_faces_b, (1, 0, 1, 1)),
    ]

    for collection, color in regions:

        for item in collection:
            face = bm.faces[item.index]

            uv_sum = Vector((0, 0))
            count = 0

            for loop in face.loops:
                uv_sum += loop[uv_layer].uv
                count += 1

            uv_center = uv_sum / count

            DEBUG_CORNER_POINTS.append(
                (uv_center.copy(), color)
            )

def clip_line_to_unit_square(pivot, direction):
    """Clip infinite line to UV square [0,1]x[0,1]."""
    points = []

    for axis in range(2):  # x, y
        for bound in (0.0, 1.0):
            if abs(direction[axis]) < 1e-6:
                continue

            t = (bound - pivot[axis]) / direction[axis]
            p = pivot + direction * t

            if 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0:
                points.append(p)

    if len(points) >= 2:
        return points[0], points[1]

    return None


# -----------------------------
# Drawing
# -----------------------------
def enable_uv_debug():

    global _uv_debug_handle

    if _uv_debug_handle is not None:
        return

    _uv_debug_handle = bpy.types.SpaceImageEditor.draw_handler_add(
        draw_uv_debug,
        (),
        'WINDOW',
        'POST_VIEW'
    )

def disable_uv_debug():

    global _uv_debug_handle

    if _uv_debug_handle is not None:

        bpy.types.SpaceImageEditor.draw_handler_remove(
            _uv_debug_handle,
            'WINDOW'
        )

        _uv_debug_handle = None

def draw_uv_debug():

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    # Normal Fold Rendering

    coords = []

    for a, b in DEBUG_UV_SEGMENTS:
        coords.extend([a, b])

    if coords:

        batch = batch_for_shader(
            shader,
            'LINES',
            {"pos": coords}
        )

        shader.bind()
        shader.uniform_float("color", (0, 1, 0, 1))

        batch.draw(shader)

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    # Leak Finder
    coords = [] 
    for a, b in DEBUG_TESTED_EDGES:
        coords.extend([a, b])

    if coords:

        batch = batch_for_shader(
            shader,
            'LINES',
            {"pos": coords}
        )

        shader.bind()
        shader.uniform_float("color", (1, 0, 0, 1))

        batch.draw(shader)

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    # Interaction Fold Rendering
    colors = get_evenly_spaced_colors(len(INTERACTION_UV_SEGMENT_LISTS))
    for index, segments in enumerate(INTERACTION_UV_SEGMENT_LISTS):
        coords = []
        for a, b in segments:
            coords.extend([a, b])

        if coords:

            batch = batch_for_shader(
                shader,
                'LINES',
                {"pos": coords}
            )

            shader.bind()
            shader.uniform_float("color", colors[index])
            gpu.state.line_width_set(len(INTERACTION_UV_SEGMENT_LISTS)*2 - (index*2))
            batch.draw(shader)

def get_evenly_spaced_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        rgba = (rgb[0], rgb[1], rgb[2], 1)
        colors.append(rgba)
    return colors


def draw_crease_pattern():
    obj = bpy.context.object
    if not obj:
        return

    lines = get_crease_lines(obj)
    if not lines:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    for line in lines:
        segment = clip_line_to_unit_square(line["pivot"], line["dir"])
        if not segment:
            continue

        p1, p2 = segment

        coords = [p1, p2]

        # Mountain vs Valley coloring
        if line["angle"] >= 0:
            color = (1.0, 0.2, 0.2, 1.0)  # red
        else:
            color = (0.2, 0.2, 1.0, 1.0)  # blue

        batch = batch_for_shader(shader, 'LINES', {"pos": coords})

        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)


# -----------------------------
# Public API
# -----------------------------

def show_crease_pattern():
    global _crease_handle

    if _crease_handle is not None:
        return  # already active

    _crease_handle = bpy.types.SpaceImageEditor.draw_handler_add(
        draw_crease_pattern,
        (),
        'WINDOW',
        'POST_VIEW'
    )


def hide_crease_pattern():
    global _crease_handle

    if _crease_handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_crease_handle, 'WINDOW')
        _crease_handle = None


def toggle_crease_pattern():
    global _crease_handle

    if _crease_handle is None:
        show_crease_pattern()
    else:
        hide_crease_pattern()


# -----------------------------
# Operator
# -----------------------------

class ORIGAMI_OT_toggle_crease_pattern(bpy.types.Operator):
    bl_idname = "origami.toggle_crease_pattern"
    bl_label = "Toggle Crease Pattern"

    def execute(self, context):
        toggle_crease_pattern()
        return {'FINISHED'}


# -----------------------------
# Registration
# -----------------------------

def register():
    bpy.utils.register_class(ORIGAMI_OT_toggle_crease_pattern)


def unregister():
    hide_crease_pattern()
    bpy.utils.unregister_class(ORIGAMI_OT_toggle_crease_pattern)