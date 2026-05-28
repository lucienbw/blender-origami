import bpy
import gpu
from mathutils import Vector
from gpu_extras.batch import batch_for_shader

from .origami_state import DEBUG_UV_SEGMENTS
from .origami_state import DEBUG_BLOCKED_EDGES
from .origami_state import DEBUG_TESTED_EDGES

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

    # -------------------------
    # STORED CREASE SEGMENTS
    # GREEN
    # -------------------------

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

    # -------------------------
    # TESTED EDGES
    # BLUE
    # -------------------------

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
        shader.uniform_float("color", (0, 0, 1, 0.3))

        batch.draw(shader)

    # -------------------------
    # BLOCKED EDGES
    # RED
    # -------------------------

    coords = []

    for a, b in DEBUG_BLOCKED_EDGES:
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