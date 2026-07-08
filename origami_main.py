import bpy
import bmesh
import math
import blf
import gpu
import json
import bpy
import numpy as np

from bpy.types import PropertyGroup, Panel, Operator
from bpy.props import FloatProperty, FloatVectorProperty, CollectionProperty, IntProperty, PointerProperty, EnumProperty, BoolProperty
from mathutils import Vector, Matrix
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
from collections import defaultdict
from math import radians

from .origami_state import HIGHLIGHT_FACES
from .origami_state import DRAW_HANDLERS
from .origami_state import DEBUG_UV_SEGMENTS
from .origami_state import INTERACTION_UV_SEGMENT_LISTS
from .origami_state import DEBUG_BLOCKED_EDGES
from .origami_state import DEBUG_TESTED_EDGES

from .origami_crease_visualizer import enable_uv_debug
from .origami_crease_visualizer import disable_uv_debug

EPS = 1e-5

DEBUG_FLOODS = []
def update_timeline(self, context):
    obj = context.object
    if not obj:
        return

    FoldEvaluator.evaluate(obj, obj.fold_timeline)

def on_interaction_type_changed(self, context):
    self.regions.clear()
    groups = context.object.fold_groups
    group = find_group_for_interaction(groups, self)
    index = next((i for i, obj in enumerate(groups) if obj == group), None)
    current_time = context.object.fold_timeline
    FoldEvaluator.evaluate(context.object, index)
    build_interaction_data_from_type(self, context)
    FoldEvaluator.evaluate(context.object, current_time)

class OrigamiFaceIndex(PropertyGroup):
    index: bpy.props.IntProperty()

class OrigamiUVSegment(PropertyGroup):
    a: FloatVectorProperty(size=2)
    b: FloatVectorProperty(size=2)

class OrigamiRegion(PropertyGroup):
    region_name: bpy.props.StringProperty(
        default="Base"
    )

    seed_uv: FloatVectorProperty(size=2)

    crease_uv_segments: CollectionProperty(
        type=OrigamiUVSegment
    )

    cached_faces: CollectionProperty(
        type=OrigamiFaceIndex
    )

class OrigamiAxis(PropertyGroup):
    value: FloatVectorProperty(size=3)

class FoldInteraction(PropertyGroup):
    owner_fold_index: IntProperty()
    other_fold_index: IntProperty()

    interaction_type: EnumProperty(
        name="Interaction",
        items=[
            ("NONE", "None", ""),
            ("CORNER_M", "Corner Mountain", ""),
            ("CORNER_V", "Corner Valley", ""),
            ("RABBIT", "Rabbit Ear", ""),
            ("COLLAPSE", "Collapse", ""),
            ("NESTED", "Nested", ""),
        ],
        default="NONE",
        update=on_interaction_type_changed
    )

    regions: CollectionProperty(
        type=OrigamiRegion
    )

    def get_region(self, name):
        for region in self.regions:
            if region.region_name == name:
                return region
        region = self.regions.add()
        region.region_name = name

        return region

    axes: CollectionProperty(
        type=OrigamiAxis
    )

    adjuster: bpy.props.FloatProperty(
        name="Angle",
        subtype='ANGLE',
        default=0.0,
        step=100,
        update=update_timeline
    )

    origin: FloatVectorProperty(size=3)

class OrigamiFold(PropertyGroup):
    pivot_3d: FloatVectorProperty(size=3)
    axis: FloatVectorProperty(size=3)
    uv_axis: PointerProperty(type=OrigamiUVSegment)
    angle: bpy.props.FloatProperty(
        name="Angle",
        subtype='ANGLE',
        default=0.0,
        step=100,
        update=update_timeline
    )
    region: PointerProperty(type=OrigamiRegion)

    selected: bpy.props.BoolProperty(
        name="Selected",
        default=False
    )
    muted: bpy.props.BoolProperty(
        name="Muted",
        default=False
    )
    interactions: CollectionProperty(
        type=FoldInteraction
    )

class FoldOperation:

    def __init__(self, region_faces, matrix):
        self.region_faces = region_faces
        self.matrix = matrix

class RabbitEar(PropertyGroup):
    fold_a_index: IntProperty()
    fold_b_index: IntProperty()
    axis_a: FloatVectorProperty(size=3)
    axis_b: FloatVectorProperty(size=3)
    center: FloatVectorProperty(size=3)

    region_a_base_faces: CollectionProperty(type=OrigamiFaceIndex)
    region_a_tip_faces: CollectionProperty(type=OrigamiFaceIndex)
    region_b_base_faces: CollectionProperty(type=OrigamiFaceIndex)
    region_b_tip_faces: CollectionProperty(type=OrigamiFaceIndex)

    selected: bpy.props.BoolProperty(
        name="Selected",
        default=False
    )

class CornerFold(PropertyGroup):
    fold_a_index: IntProperty()
    fold_b_index: IntProperty()
    axis_a: FloatVectorProperty(size=3)
    axis_b: FloatVectorProperty(size=3)
    pivot: FloatVectorProperty(size=3)

    region_a_base_faces: CollectionProperty(type=OrigamiFaceIndex)
    region_b_base_faces: CollectionProperty(type=OrigamiFaceIndex)
    region_a_corner_faces: CollectionProperty(type=OrigamiFaceIndex)
    region_b_corner_faces: CollectionProperty(type=OrigamiFaceIndex)

    selected: bpy.props.BoolProperty(
        name="Selected",
        default=False
    )

class OrigamiFoldGroup(PropertyGroup):
    name: bpy.props.StringProperty(default="Fold Group")
    folds: CollectionProperty(type=OrigamiFold)

class PreviewFold:
    def __init__(self, pivot_3d, axis, angle, region_faces):
        self.pivot_3d = pivot_3d
        self.axis = axis
        self.angle = angle
        self.region_faces = region_faces

class ORIGAMI_UL_fold_groups(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        group = item
        layout.label(text=f"Group {index} ({len(group.folds)} folds)", icon='GROUP')

class ORIGAMI_UL_folds(bpy.types.UIList):

    def draw_item(
        self,
        context,
        layout,
        data,
        item,
        icon,
        active_data,
        active_propname,
        index
    ):
        fold = item

        row = layout.row(align=True)

        row.prop(
            fold,
            "selected",
            text=""
        )

        row.prop(
            fold,
            "muted",
            text="",
            icon='HIDE_ON' if fold.muted else 'HIDE_OFF',
            emboss=False
        )

        split = row.split(factor=0.6)

        split.label(
            text=f"{len(fold.region.cached_faces)} faces",
            icon='MOD_SIMPLEDEFORM'
        )

        split.prop(
            fold,
            "angle",
            text=""
        )

class ORIGAMI_UL_fold_interaction(bpy.types.UIList):

    def draw_item(
        self,
        context,
        layout,
        data,
        item,
        icon,
        active_data,
        active_propname,
        index
    ):
        interaction = item

        row = layout.row(align=True)

        row.prop(
            interaction,
            "selected",
            text=""
        )

        row.label(
            text=f"Type: {interaction.interaction_type} | Fold: {interaction.other_fold_index}",
            icon='MOD_SIMPLEDEFORM'
        )

class CreasePathNode:

    _id_counter = 0

    def __init__(self, co, kind, vert=None, edge=None):

        self.id = CreasePathNode._id_counter
        CreasePathNode._id_counter += 1

        self.co = co
        self.kind = kind
        self.vert = vert
        self.edge = edge

class CreasePathSegment:
    def __init__(self, a, b, existing_edge=None):
        self.a = a
        self.b = b
        self.existing_edge = existing_edge

    @property
    def is_existing(self):
        return self.existing_edge is not None

class CreasePath:
    def __init__(self):
        self.nodes = []
        self.segments = []

def undo_post_handler(dummy):
    for obj in bpy.data.objects:
        if hasattr(obj, "fold_timeline"):
            None
            #Can use this later to implement proper undo

def clean_up_interactions(self, context):
    for idx, group in enumerate(context.object.fold_groups):
        print(f"GROUP {idx} ----------")
        for f, fold in enumerate(group.folds):
            print(f"fold {f}: axis-{fold.axis}, angle-{fold.angle}, faces-{[item.index for item in fold.region.cached_faces]}")
            for i in range(len(fold.interactions) - 1, -1, -1):
                interaction = fold.interactions[i]

                if interaction.other_fold_index >= len(group.folds):
                    fold.interactions.remove(i)

def on_active_fold_changed(self, context):
    ensure_draw_handler()
    obj = context.object
    if not obj:
        return

    HIGHLIGHT_FACES.clear()
    DEBUG_UV_SEGMENTS.clear()
    DEBUG_TESTED_EDGES.clear()
    INTERACTION_UV_SEGMENT_LISTS.clear()

    g = obj.active_fold_group
    f = obj.active_fold

    if g >= len(obj.fold_groups):
        return

    group = obj.fold_groups[g]

    if f >= len(group.folds):
        return

    fold = group.folds[f]

    absolute_fold_index = 0
    for i in range(g):
        absolute_fold_index += len(obj.fold_groups[i].folds)
    absolute_fold_index += f

    interactions = get_interactions_involving_fold(group, f)
    #for edge in DEBUG_FLOODS[absolute_fold_index]:
    #    DEBUG_TESTED_EDGES.append(edge)

    if len(interactions) == 0 or interactions[0].interaction_type == 'NONE':
        for uv_segment in fold.region.crease_uv_segments:
            DEBUG_UV_SEGMENTS.append((uv_segment.a, uv_segment.b))
    else:
        for interaction in interactions:
            for region in interaction.regions:
                new_segment_list = []
                for uv_segment in region.crease_uv_segments:
                    new_segment_list.append((tuple(uv_segment.a), tuple(uv_segment.b)))
                INTERACTION_UV_SEGMENT_LISTS.append(new_segment_list)

    for rf in fold.region.cached_faces:
        HIGHLIGHT_FACES.append(rf.index)

    obj["_active_preview_fold"] = {
        "pivot": list(fold.pivot_3d),
        "axis": list(fold.axis)
    }

    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

def on_active_group_changed(self, context):
    obj = context.object
    if not obj:
        return

    g = obj.active_fold_group
    obj.active_fold = 0

    if g >= len(obj.fold_groups):
        return

    for group in obj.fold_groups:
        if group != obj.fold_groups[g]:
            for fold in group.folds:
                fold.selected = False

def initialize_region_from_faces(obj, bm, region, face_indices, region_name="Base", seed_mode="first_face"):

    ensure_full_lookup_table(bm)

    face_indices = set(face_indices)

    if not face_indices:
        raise ValueError("Cannot initialize region from empty face set")

    uv_layer = bm.loops.layers.uv.active

    if uv_layer is None:
        raise RuntimeError("Mesh has no active UV layer")

    region.region_name = region_name
    region.crease_uv_segments.clear()
    region.cached_faces.clear()

    if seed_mode == "first_face":

        seed_face_index = next(iter(face_indices))
        seed_face = bm.faces[seed_face_index]

        uv_sum = Vector((0.0, 0.0))

        for loop in seed_face.loops:
            uv_sum += loop[uv_layer].uv

        region.seed_uv = uv_sum / len(seed_face.loops)

    elif seed_mode == "centroid":

        uv_sum = Vector((0.0, 0.0))
        count = 0

        for f_idx in face_indices:
            f = bm.faces[f_idx]

            for loop in f.loops:
                uv_sum += loop[uv_layer].uv
                count += 1

        region.seed_uv = uv_sum / count

    else:
        raise ValueError(
            f"Unknown seed_mode '{seed_mode}'"
        )

    seen_edges = set()

    for f_idx in face_indices:

        face = bm.faces[f_idx]

        for loop in face.loops:

            edge = loop.edge

            if edge in seen_edges:
                continue

            seen_edges.add(edge)

            is_boundary = False

            for linked_face in edge.link_faces:

                if linked_face.index not in face_indices:
                    is_boundary = True
                    break

            if len(edge.link_faces) == 1:
                is_boundary = True

            if not is_boundary:
                continue

            uv1 = loop[uv_layer].uv.copy()
            uv2 = loop.link_loop_next[uv_layer].uv.copy()

            seg = region.crease_uv_segments.add()
            seg.a = uv1
            seg.b = uv2
    for f_idx in face_indices:
        item = region.cached_faces.add()
        item.index = f_idx

def build_group_operations(obj, bm, group, partial, preview_fold=None):

    operations = []

    handled_folds = set()
    handled_interactions = set()
    used_faces_dict = {i: set() for i in range(len(group.folds))}

    for fold_index, fold in enumerate(group.folds):
        interaction_folds = get_interactions_involving_fold(group, fold_index)
        for interaction in interaction_folds:
            
            owner_index = interaction.owner_fold_index
            other_index = interaction.other_fold_index

            if getattr(group.folds[owner_index], "muted", False) or getattr(group.folds[other_index], "muted", False):
                continue
            
            if interaction in handled_interactions:
                continue

            handled_folds.add(owner_index)
            handled_folds.add(other_index)
            handled_interactions.add(interaction)

            match interaction.interaction_type:

                case "RABBIT":

                    evals = rabbit_ear_eval(
                        group,
                        interaction,
                        partial,
                        bm
                    )

                    operations.extend([
                        FoldOperation(
                            [item.index for item in interaction.get_region("tip_A").cached_faces],
                            evals["a_tip_faces"]
                        ),
                        FoldOperation(
                            [item.index for item in interaction.get_region("tip_B").cached_faces],
                            evals["b_tip_faces"]
                        ),
                        FoldOperation(
                            [item.index for item in interaction.get_region("base_A").cached_faces],
                            evals["a_base_faces"]
                        ),
                        FoldOperation(
                            [item.index for item in interaction.get_region("base_B").cached_faces],
                            evals["b_base_faces"]
                        ),
                    ])

                case "CORNER_M":

                    evals = corner_fold_eval(
                        group,
                        1,
                        interaction,
                        partial,
                        bm
                    )

                    operations.extend([
                        FoldOperation(
                            [item.index for item in interaction.get_region("corner_faces_a").cached_faces],
                            evals["a_corner_faces"]
                        ),
                        FoldOperation(
                            [item.index for item in interaction.get_region("corner_faces_b").cached_faces],
                            evals["b_corner_faces"]
                        ),
                    ])
                    
                case "CORNER_V":

                    evals = corner_fold_eval(
                        group,
                        -1,
                        interaction,
                        partial,
                        bm
                    )

                    operations.extend([
                        FoldOperation(
                            [item.index for item in interaction.get_region("corner_faces_a").cached_faces],
                            evals["a_corner_faces"]
                        ),
                        FoldOperation(
                            [item.index for item in interaction.get_region("corner_faces_b").cached_faces],
                            evals["b_corner_faces"]
                        ),
                    ])

                case "COLLAPSE":

                    evals = collapse_fold_eval(
                        group,
                        interaction,
                        partial,
                        bm
                    )

                    operations.extend([

                        FoldOperation(
                            [item.index for item in interaction.get_region("tip_A").cached_faces],
                            evals["a_tip_faces"]
                        ),
                        FoldOperation(
                            [item.index for item in interaction.get_region("tip_B").cached_faces],
                            evals["b_tip_faces"]
                        ),
                        FoldOperation(
                            [item.index for item in interaction.get_region("top").cached_faces],
                            evals["top_faces"]
                        ),
                    ])

                case "NESTED":

                    evals = nested_fold_eval(
                        group,
                        interaction,
                        partial,
                        bm
                    )
                    interaction.get_region("child").cached_faces
                    operations.extend([
                        FoldOperation(
                            [item.index for item in interaction.get_region("child").cached_faces],
                            evals["child_faces"]
                        ),
                    ])

            for op in operations:
                used_faces_dict[owner_index].update(op.region_faces)
                used_faces_dict[other_index].update(op.region_faces)

    group_folds = list(group.folds)

    if preview_fold:
        group_folds = [preview_fold]

    for fold_index, fold in enumerate(group_folds):
        
        if isinstance(fold, PreviewFold):
            region_faces = fold.region_faces
        else:
            if getattr(fold, "muted", False):
                continue

            fold_faces = {
                f.index
                for f in fold.region.cached_faces
            }

            remaining_faces = (
                fold_faces
                - used_faces_dict[fold_index]
            )
            region_faces = list(remaining_faces)

            if fold_index in handled_folds and not remaining_faces:
                continue

        angle = fold.angle * partial

        rot = fold_matrix(
            Vector(fold.pivot_3d),
            Vector(fold.axis).normalized(),
            angle
        )

        operations.append(
            FoldOperation(
                region_faces,
                rot
            )
        )

    return operations

def find_group_for_interaction(groups, target_interaction):
    for group in groups:
        for fold in group.folds:
            for interaction in fold.interactions:
                if interaction == target_interaction:
                    return group
    return None

def find_interaction(group, owner_index, other_index):

    for interaction in group.folds[owner_index].interactions:
        if interaction.other_fold_index == other_index:
            return interaction

    return None

def ensure_interaction(group, owner_index, other_index):
    interaction = find_interaction(group, owner_index, other_index)

    if interaction:
        return interaction

    interaction = group.folds[owner_index].interactions.add()
    interaction.owner_fold_index = owner_index
    interaction.other_fold_index = other_index

    return interaction

def get_interactions_involving_fold(group, index):
    interaction_acc = []
    for index, fold in enumerate(group.folds):
        for interaction in fold.interactions:
            if interaction.other_fold_index > index:
                interaction_acc.append(interaction)

    return interaction_acc

def build_interaction_data_from_type(interaction, context):
    obj = context.object
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    group = find_group_for_interaction(obj.fold_groups, interaction)
    match interaction.interaction_type:
        case "RABBIT":
            build_rabbit_interaction_data(obj, bm, group, interaction, True)
        case "CORNER_M":
            build_corner_interaction_data(obj, bm, group, interaction)
        case "CORNER_V":
            build_corner_interaction_data(obj, bm, group, interaction)
        case "COLLAPSE":
            build_collapse_interaction_data(obj, bm, group, interaction)
        case "NESTED":
            build_nested_interaction_data(obj, bm, group, interaction)

def interactions_for_fold(group, fold_index):
    result = []

    for owner_index, fold in enumerate(group.folds):
        for interaction in fold.interactions:

            if owner_index == fold_index:
                result.append(interaction)

            elif interaction.other_fold_index == fold_index:
                result.append(interaction)

    return result

def shares_face(fold_a, fold_b):
    faces_a = {face.index for face in fold_a.region.cached_faces}
    faces_b = {face.index for face in fold_b.region.cached_faces}

    return not faces_a.isdisjoint(faces_b)

def check_new_overlapping_folds(self, context, group_index, new_fold_index):
    obj = context.object
    group = obj.fold_groups[group_index]
    new_fold = group.folds[new_fold_index]
    for index, fold in enumerate(group.folds):
        if index != new_fold_index:
            if (shares_face(fold, new_fold)):
                ensure_interaction(group, index, new_fold_index)

def closest_point_on_edge(p, a, b):
    ab = b - a
    denom = ab.dot(ab)

    if denom < 1e-12:
        return a.copy()

    t = (p - a).dot(ab) / denom
    t = max(0.0, min(1.0, t))

    return a + ab * t

def uv_equal(a, b):
    return (a - b).length <= EPS

def point_segment_distance(p, a, b):

    ab = b - a

    if ab.length_squared < 1e-12:
        return (p - a).length

    t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))

    closest = a + ab * t

    return (p - closest).length

def edge_lies_on_crease(e1, e2, crease_segments):

    def project_scalar(p, origin, direction):
        return (p - origin).dot(direction)

    ANGLE_EPS = 0.95

    edge_dir = (e2 - e1)
    edge_len = edge_dir.length

    DIST_EPS = 0.05 * edge_len

    if edge_len < 1e-8:
        return False

    edge_dir.normalize()

    for seg in crease_segments:

        a = Vector(seg.a)
        b = Vector(seg.b)

        seg_dir = b - a
        seg_len = seg_dir.length

        if seg_len < 1e-8:
            continue

        seg_dir.normalize()

        if abs(edge_dir.dot(seg_dir)) < ANGLE_EPS:
            continue

        d1 = point_segment_distance(e1, a, b)
        d2 = point_segment_distance(e2, a, b)

        if max(d1, d2) > DIST_EPS:
            continue

        proj1 = project_scalar(e1, a, seg_dir)
        proj2 = project_scalar(e2, a, seg_dir)

        seg_min = 0
        seg_max = seg_len

        edge_min = min(proj1, proj2)
        edge_max = max(proj1, proj2)

        overlap = min(edge_max, seg_max) - max(edge_min, seg_min)

        if overlap > -DIST_EPS:
            return True
    #DEBUG_TESTED_EDGES.append((e1,e2))
    return False

def crosses_crease_uv(bm, f1_idx, f2_idx, uv_layer, crease_uv_segments):

    edge_uvs = shared_uv_edge_between_faces(
        bm,
        f1_idx,
        f2_idx,
        uv_layer
    )

    if edge_uvs is None:
        return False


    e1, e2 = edge_uvs

    return edge_lies_on_crease(
        e1,
        e2,
        crease_uv_segments
    )

def shared_uv_edge_between_faces(bm, f1_idx, f2_idx, uv_layer):

    f1 = bm.faces[f1_idx]
    f2 = bm.faces[f2_idx]

    shared_edges = set(f1.edges) & set(f2.edges)
    if len(shared_edges) < 1:
        return None

    if not shared_edges:
        return None

    edge = next(iter(shared_edges))

    for loop in f1.loops:
        if loop.edge == edge:

            uv1 = loop[uv_layer].uv.copy()
            uv2 = loop.link_loop_next[uv_layer].uv.copy()

            return uv1, uv2

    return None

def uv_curve_flood_fill(bm, seed_idx, adjacency, uv_layer, crease_uv_segments):
    visited = set()
    stack = [seed_idx]
    ensure_full_lookup_table(bm)
    DEBUG_TESTED_EDGES.clear()
    while stack:
        f_idx = stack.pop()

        if f_idx in visited:
            continue

        visited.add(f_idx)

        for n in adjacency.get(f_idx, []):

            if n in visited:
                continue

            if crosses_crease_uv(bm, f_idx, n, uv_layer, crease_uv_segments):
                continue 

            stack.append(n)

    #DEBUG_FLOODS.append(DEBUG_TESTED_EDGES.copy())

    return visited

def rebuild_regions_from_seed_uv(obj, bm, group=None):
    DEBUG_FLOODS.clear()
    uv_layer = bm.loops.layers.uv.active

    adjacency = build_face_graph(bm, set())
    
    for g in obj.fold_groups:
        if group != None and g != group:
            continue 
        for fold in g.folds:

            seed_face = find_face_from_seed_uv(bm, uv_layer, fold.region.seed_uv)
            if not seed_face:
                print("NO SEED FACE")
                continue
            region = uv_curve_flood_fill(
                bm,
                seed_face.index,
                adjacency,
                uv_layer,
                fold.region.crease_uv_segments
            )

            fold.region.cached_faces.clear()
            for f_idx in region:
                item = fold.region.cached_faces.add()
                item.index = f_idx

            for interaction in fold.interactions:
                for region in interaction.regions:
                    seed_face = find_face_from_seed_uv(bm, uv_layer, region.seed_uv)
                    if not seed_face:
                        print("NO SEED FACE")
                        continue
                    region_faces = uv_curve_flood_fill(
                        bm,
                        seed_face.index,
                        adjacency,
                        uv_layer,
                        region.crease_uv_segments
                    )

                    region.cached_faces.clear()
                    for f_idx in region_faces:
                        item = region.cached_faces.add()
                        item.index = f_idx
                    print(f"region {region.region_name} has {len(region.cached_faces)} faces")
    
def find_face_from_seed_uv(bm, uv_layer, seed_uv):
    seed_uv = Vector(seed_uv)

    best_face = None
    best_dist = 1e10

    for f in bm.faces:
        uv_sum = Vector((0.0, 0.0))
        n = 0

        for l in f.loops:
            uv_sum += l[uv_layer].uv
            n += 1

        if n == 0:
            continue

        center_uv = uv_sum / n
        d = (center_uv - seed_uv).length

        if d < best_dist:
            best_dist = d
            best_face = f

    return best_face
        
def classify_face(f, fold):
    center = f.calc_center_median()
    d = (center - fold.pivot).dot(fold_plane_normal)
    return d > 0
    
def closest_point_between_lines(p1, d1, p2, d2):
    """
    Returns midpoint of shortest segment between two lines.
    This is the best "intersection" approximation.
    """

    d1 = d1.normalized()
    d2 = d2.normalized()

    r = p1 - p2

    a = d1.dot(d1)
    b = d1.dot(d2)
    c = d2.dot(d2)
    d = d1.dot(r)
    e = d2.dot(r)

    denom = a * c - b * b

    if abs(denom) < 1e-8:
        return (p1 + p2) * 0.5

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom

    c1 = p1 + d1 * t
    c2 = p2 + d2 * s

    return (c1 + c2) * 0.5

def depsgraph_handler(scene, depsgraph):
    for update in depsgraph.updates:
        obj = update.id.original

        if not isinstance(obj, bpy.types.Object):
            continue

        if obj.type != 'MESH' or obj.mode != 'EDIT':
            continue

        if obj.get("_origami_is_evaluating", False):
            continue

        if not update.is_updated_geometry:
            continue

        me = obj.data

        bm = bmesh.from_edit_mesh(me)

        if obj.get("_topo_vert_len", 0) != len(bm.verts) or obj.get("_topo_edge_len", 0) != len(bm.edges) or obj.get("_topo_face_len", 0) != len(bm.faces):
            print("updating dirty mesh topology")
            obj["_topo_vert_len"] = len(bm.verts)
            obj["_topo_edge_len"] = len(bm.edges)
            obj["_topo_face_len"] = len(bm.faces)
            rebuild_regions_from_seed_uv(obj, bm)

def origami_frame_update(scene):

    for obj in scene.objects:

        if not hasattr(obj, "fold_timeline"):
            continue

        FoldEvaluator.evaluate(obj, obj.fold_timeline)

def collect_sheet_region(bm, seed_face, crease_angle=radians(25)):

    region = set()
    stack = [seed_face]

    while stack:

        face = stack.pop()

        if face in region:
            continue

        region.add(face)

        for edge in face.edges:

            if len(edge.link_faces) != 2:
                continue

            f1, f2 = edge.link_faces

            other = f2 if f1 == face else f1

            angle = face.normal.angle(other.normal)

            if angle > crease_angle:
                continue

            if other not in region:
                stack.append(other)

    return list(region)

def apply_crease_path(bm, path, threshold=1e-6):

    ensure_full_lookup_table(bm)

    node_to_vert = {}
    new_verts = []

    for node in path.nodes:
        if node.kind == "EXISTING_VERT" and node.vert and node.vert.is_valid:
            node_to_vert[node.id] = node.vert

    edge_lookup = {}

    for e in bm.edges:
        v1, v2 = e.verts
        key = tuple(sorted((v1.index, v2.index)))
        edge_lookup[key] = e

    edge_splits = {}

    def edge_key_from_node(node):
        if not node.edge:
            return None
        v1, v2 = node.edge.verts
        return tuple(sorted((v1.index, v2.index)))

    for node in path.nodes:

        if node.kind != "NEW_VERT":
            continue

        edge = node.edge
        if not edge:
            continue

        v1, v2 = edge.verts
        key = tuple(sorted((v1.index, v2.index)))

        real_edge = edge_lookup.get(key)
        if not real_edge:
            continue

        edge_len = (v2.co - v1.co).length
        if edge_len < 1e-8:
            continue

        t = (node.co - v1.co).length / edge_len
        t = max(0.0, min(1.0, t))

        edge_splits.setdefault(key, []).append((t, node))

    for key, splits in edge_splits.items():

        edge = edge_lookup.get(key)
        if not edge or not edge.is_valid:
            continue

        splits.sort(key=lambda x: x[0])

        current_edge = edge
        current_start = edge.verts[0]
        previous_t = 0.0

        for t, node in splits:

            local_t = (t - previous_t) / (1.0 - previous_t)
            local_t = max(0.0001, min(0.9999, local_t))

            try:
                new_edge, new_vert = bmesh.utils.edge_split(
                    current_edge,
                    current_start,
                    local_t
                )
            except:
                continue

            new_vert.co = node.co.copy()
            node_to_vert[node.id] = new_vert
            new_verts.append(new_vert)

            current_edge = new_edge
            current_start = new_vert
            previous_t = t

    ensure_full_lookup_table(bm)

    for seg in path.segments:

        v1 = node_to_vert.get(seg.a.id)
        v2 = node_to_vert.get(seg.b.id)

        if not v1 or not v2:
            continue

        if not v1.is_valid or not v2.is_valid:
            continue

        if v1 == v2:
            continue

        shared_faces = [f for f in v1.link_faces if v2 in f.verts]

        for face in shared_faces:
            if not face.is_valid:
                continue
            try:
                bmesh.utils.face_split(face, v1, v2)
                break
            except:
                pass

    ensure_full_lookup_table(bm)
    return new_verts

def signed_plane_distance(p, plane_co, plane_no):
    return (p - plane_co).dot(plane_no)

def build_crease_path(bm, candidate_faces, plane_origin, plane_normal, threshold):

    ensure_full_lookup_table(bm)

    path = CreasePath()

    candidate_face_set = set(candidate_faces)

    candidate_edges = set()
    for face in candidate_face_set:
        candidate_edges.update(face.edges)

    vert_nodes = {}
    edge_nodes = {}

    edge_hits = {}

    def edge_key(v1, v2):
        a, b = sorted((v1.index, v2.index))
        return (a, b)

    for edge in candidate_edges:

        v1, v2 = edge.verts

        d1 = signed_plane_distance(v1.co, plane_origin, plane_normal)
        d2 = signed_plane_distance(v2.co, plane_origin, plane_normal)

        key = edge_key(v1, v2)

        if abs(d1) < threshold and abs(d2) < threshold:

            if v1 not in vert_nodes:
                n1 = CreasePathNode(v1.co.copy(), "EXISTING_VERT", vert=v1)
                vert_nodes[v1] = n1
                path.nodes.append(n1)

            if v2 not in vert_nodes:
                n2 = CreasePathNode(v2.co.copy(), "EXISTING_VERT", vert=v2)
                vert_nodes[v2] = n2
                path.nodes.append(n2)

            a = vert_nodes[v1]
            b = vert_nodes[v2]

            path.segments.append(CreasePathSegment(a, b, existing_edge=key))
            edge_hits[key] = (a, b)
            continue

        if d1 * d2 < 0:

            t = d1 / (d1 - d2)
            p = v1.co.lerp(v2.co, t)

            if (p - v1.co).length < threshold:
                node = vert_nodes.get(v1)
                if node is None:
                    node = CreasePathNode(v1.co.copy(), "EXISTING_VERT", vert=v1)
                    vert_nodes[v1] = node
                    path.nodes.append(node)

            elif (p - v2.co).length < threshold:
                node = vert_nodes.get(v2)
                if node is None:
                    node = CreasePathNode(v2.co.copy(), "EXISTING_VERT", vert=v2)
                    vert_nodes[v2] = node
                    path.nodes.append(node)

            else:
                node = edge_nodes.get(key)
                if node is None:
                    node = CreasePathNode(p.copy(), "NEW_VERT", edge=edge)
                    edge_nodes[key] = node
                    path.nodes.append(node)

            edge_hits[key] = node

    for face in candidate_face_set:

        hits = []

        for edge in face.edges:

            v1, v2 = edge.verts
            key = edge_key(v1, v2)

            hit = edge_hits.get(key)
            if hit is None:
                continue

            if isinstance(hit, tuple):
                continue

            hits.append(hit)

        if len(hits) != 2:
            continue

        a, b = hits

        if a == b:
            continue

        path.segments.append(CreasePathSegment(a, b))

    return path

def reflect_vector_about_axis(v, axis):
    axis = axis.normalized()
    proj = axis * v.dot(axis)
    return 2 * proj - v

def find_shared_edge_between_regions(bm, region_a, region_b):

    faces_a = {bm.faces[i] for i in region_a}
    faces_b = {bm.faces[i] for i in region_b}

    candidate_edges = []

    for edge in bm.edges:

        linked = set(edge.link_faces)

        has_a = bool(linked & faces_a)
        has_b = bool(linked & faces_b)

        if has_a and has_b:
            candidate_edges.append(edge)

    if not candidate_edges:
        return None

    return candidate_edges[0]

def angle_bisector(axis_a, axis_b):
    a = Vector(axis_a).normalized()
    b = Vector(axis_b).normalized()

    # Handle opposing directions
    if a.dot(b) < 0:
        b = -b

    bisector = a + b

    if bisector.length < 1e-6:
        return None

    return bisector.normalized()

def build_corner_interaction_data(obj, bm, group, interaction):

    foldA = group.folds[interaction.owner_fold_index]
    foldB = group.folds[interaction.other_fold_index]

    facesA = {f.index for f in foldA.region.cached_faces}
    facesB = {f.index for f in foldB.region.cached_faces}

    overlap = facesA & facesB

    if not overlap:
        return None

    onlyA = facesA - overlap
    onlyB = facesB - overlap

    axisA = Vector(foldA.axis).normalized()
    axisB = Vector(foldB.axis).normalized()

    p1 = Vector(foldA.pivot_3d)
    p2 = Vector(foldB.pivot_3d)

    origin = closest_point_between_lines(
        p1,
        axisA,
        p2,
        axisB
    )
    print("origin: ",origin)

    centroidA = region_centroid(bm, list(onlyA))
    centroidB = region_centroid(bm, list(onlyB))

    dirA = (centroidA - origin)
    dirB = (centroidB - origin)

    if dirA.length < 1e-6 or dirB.length < 1e-6:
        print("Corner fold failed: degenerate centroids")
        return None

    dirA.normalize()
    dirB.normalize()

    split_dir = dirA - dirB

    if split_dir.length < 1e-6:
        print("Corner fold failed: split direction collapsed")
        return None

    split_dir.normalize()

    bisector = axisA + axisB
    up_down = bisector.cross(split_dir)
    real_split_dir = up_down.cross(bisector)

    side_a = []
    side_b = []

    eps = 0.0001 * obj.scale.length

    for f_idx in overlap:
        center = bm.faces[f_idx].calc_center_median()

        d = (center - origin).dot(real_split_dir)

        if d >= -eps:
            side_a.append(f_idx)
        else:
            side_b.append(f_idx)

    if axisA.dot(axisB) > 0:
        axisB = -axisB

    new_axisA = interaction.axes.add()
    new_axisA.value = axisA
    new_axisB = interaction.axes.add()
    new_axisB.value = axisB

    interaction.origin = origin
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, list(onlyA), "base_faces_a")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, list(onlyB), "base_faces_b")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, side_a, "corner_faces_a")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, side_b, "corner_faces_b")

def build_collapse_interaction_data(obj, bm, group, interaction):

    def classify_face_for_collapse(c, origin, nA, nMid, nB, eps):
        dA = (c - origin).dot(nA)
        dMid = (c - origin).dot(nMid)
        dB = (c - origin).dot(nB)

        if dA >= -eps and dMid >= -eps:
            return "base_A"
        elif dA < -eps and dMid >= -eps:
            return "tip_A"
        elif dMid < -eps and dB >= -eps:
            return "tip_B"
        else:
            return "base_B"
    
    foldA = group.folds[interaction.owner_fold_index]
    foldB = group.folds[interaction.other_fold_index]

    facesA = {f.index for f in foldA.region.cached_faces}
    facesB = {f.index for f in foldB.region.cached_faces}

    first_face_index = foldA.region.cached_faces[0].index
    paper_normal = bm.faces[first_face_index].normal

    overlap = facesA & facesB
    total_faces = facesA | facesB

    if not overlap:
        return None

    onlyA = facesA - overlap
    onlyB = facesB - overlap

    a = Vector(foldA.axis).normalized()
    b = Vector(foldB.axis).normalized()
    p1 = Vector(foldA.pivot_3d)
    p2 = Vector(foldB.pivot_3d)
    origin = closest_point_between_lines(p1, a, p2, b)

    centroidA = Vector(region_centroid(bm, [
        rf.index for rf in foldA.region.cached_faces
    ]))

    centroidB = Vector(region_centroid(bm, [
        rf.index for rf in foldB.region.cached_faces
    ]))

    dirA = (centroidA - origin).normalized()
    dirB = (centroidB - origin).normalized()

    v = dirA - dirB
    if v.length < 1e-6 or not all(math.isfinite(c) for c in v):
        print("ERROR: Both directions for centroids are the same")
        return None
    ensure_full_lookup_table(bm)
    if a.dot(b) > 0:
        b = -b
    split_dir = (dirA - dirB).normalized()

    axisA = Vector(foldA.axis).normalized()
    axisB = Vector(foldB.axis).normalized()

    reflected_normalA = reflect_vector_about_axis(split_dir, axisA).normalized()
    reflected_normalB = reflect_vector_about_axis(split_dir, axisB).normalized()

    working_faces = total_faces

    overlap_faces = [bm.faces[i] for i in total_faces if i < len(bm.faces)]
    candidate_faces = [
        bm.faces[i]
        for i in total_faces
        if i < len(bm.faces)
    ]
    overlap_edges = set()
    for f in overlap_faces:
        overlap_edges.update(f.edges)
    geom = list(overlap_faces) + list(overlap_edges)

    facesA = {f.index for f in foldA.region.cached_faces}
    facesB = {f.index for f in foldB.region.cached_faces}
    working_faces = onlyA | onlyB

    eps = 0.001 * obj.scale.length
    
    new_regions = {
        "base_A": [],
        "tip_A": [],
        "tip_B": [],
        "base_B": [],
        "top": []
    }
    for f in overlap:
        new_regions["top"].append(f)
    for f in working_faces:
        c = bm.faces[f].calc_center_median()
        region = classify_face_for_collapse(c, origin, reflected_normalA, split_dir, reflected_normalB, eps)
        new_regions[region].append(f)

    interaction.axes.clear()

    new_axis = interaction.axes.add()
    new_axis.value = split_dir

    interaction.origin = origin

    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, new_regions["base_A"], "base_A")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, new_regions["base_B"], "base_B")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, new_regions["tip_A"], "tip_A")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, new_regions["tip_B"], "tip_B")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, new_regions["top"], "top")

def build_nested_interaction_data(obj, bm, group, interaction):
    
    foldA = group.folds[interaction.owner_fold_index]
    foldB = group.folds[interaction.other_fold_index]

    facesA = {f.index for f in foldA.region.cached_faces}
    facesB = {f.index for f in foldB.region.cached_faces}

    overlap = facesA & facesB

    if not overlap:
        return None

    onlyA = facesA - overlap
    onlyB = facesB - overlap

    parent_faces = onlyA
    child_faces = overlap
    if len(onlyA) == 0:
        parent_faces = onlyB
    
    new_regions = {
        "parent": [],
        "child": [],
    }
    for f in parent_faces:
        new_regions["parent"].append(f)
    for f in child_faces:
        new_regions["child"].append(f)

    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, new_regions["parent"], "parent")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, new_regions["child"], "child")

def get_rabbit_bisector_plane(foldA, foldB, bm):

    a = Vector(foldA.axis).normalized()
    b = Vector(foldB.axis).normalized()

    bis = (a + b)
    if bis.length < 1e-6:
        base_normal = compute_paper_normal(bm)
        bis = a.cross(base_normal)

    bis.normalize()

    p1 = Vector(foldA.pivot_3d)
    p2 = Vector(foldB.pivot_3d)

    origin = closest_point_between_lines(p1, a, p2, b)

    plane_normal = bis

    return origin, plane_normal

def build_rabbit_interaction_data(obj, bm, group, interaction, cut):

    def classify_face_for_rabbit(c, origin, nA, nMid, nB, eps):
        dA = (c - origin).dot(nA)
        dMid = (c - origin).dot(nMid)
        dB = (c - origin).dot(nB)

        if dA >= -eps and dMid >= -eps:
            return "base_A"
        elif dA < -eps and dMid >= -eps:
            return "tip_A"
        elif dMid < -eps and dB >= -eps:
            return "tip_B"
        else:
            return "base_B"
    
    foldA = group.folds[interaction.owner_fold_index]
    foldB = group.folds[interaction.other_fold_index]

    facesA = {f.index for f in foldA.region.cached_faces}
    facesB = {f.index for f in foldB.region.cached_faces}

    first_face_index = foldA.region.cached_faces[0].index
    paper_normal = bm.faces[first_face_index].normal

    overlap = facesA & facesB
    total_faces = facesA | facesB

    if not overlap:
        return None

    onlyA = facesA - overlap
    onlyB = facesB - overlap

    axisA = Vector(foldA.axis).normalized()
    axisB = Vector(foldB.axis).normalized()
    p1 = Vector(foldA.pivot_3d)
    p2 = Vector(foldB.pivot_3d)
    origin = closest_point_between_lines(p1, axisA, p2, axisB)

    centroidA = Vector(region_centroid(bm, [
        rf.index for rf in foldA.region.cached_faces
    ]))

    centroidB = Vector(region_centroid(bm, [
        rf.index for rf in foldB.region.cached_faces
    ]))

    dirA = (centroidA - origin).normalized()
    dirB = (centroidB - origin).normalized()

    v = dirA - dirB
    if v.length < 1e-6 or not all(math.isfinite(c) for c in v):
        print("ERROR: Both directions for centroids are the same")
        return None
    ensure_full_lookup_table(bm)
    if axisA.dot(axisB) > 0:
        axisB = -axisB
    split_dir = (dirA - dirB).normalized()

    reflected_normalA = reflect_vector_about_axis(split_dir, axisA).normalized()
    reflected_normalB = reflect_vector_about_axis(split_dir, axisB).normalized()

    working_faces = total_faces

    overlap_faces = [bm.faces[i] for i in total_faces if i < len(bm.faces)]
    candidate_faces = [
        bm.faces[i]
        for i in total_faces
        if i < len(bm.faces)
    ]
    overlap_edges = set()
    for f in overlap_faces:
        overlap_edges.update(f.edges)
    geom = list(overlap_faces) + list(overlap_edges)

    threshold = 0.005 * obj.scale.length

    pathA = build_crease_path(
        bm,
        candidate_faces,
        origin,
        reflected_normalA,
        threshold
    )

    pathB = build_crease_path(
        bm,
        candidate_faces,
        origin,
        reflected_normalB,
        threshold
    )

    pathMid = build_crease_path(
        bm,
        candidate_faces,
        origin,
        split_dir,
        threshold
    )

    if cut:
        apply_crease_path(bm, pathA, threshold)
        apply_crease_path(bm, pathB, threshold)
        apply_crease_path(bm, pathMid, threshold)

    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)

    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    ensure_full_lookup_table(bm)
    rebuild_regions_from_seed_uv(obj, bm, group)
    facesA = {f.index for f in foldA.region.cached_faces}
    facesB = {f.index for f in foldB.region.cached_faces}
    working_faces = facesA | facesB

    eps = 0.001 * obj.scale.length
    
    rabbit_regions = {
        "base_A": [],
        "tip_A": [],
        "tip_B": [],
        "base_B": []
    }
    for f in working_faces:
        c = bm.faces[f].calc_center_median()
        region = classify_face_for_rabbit(c, origin, reflected_normalA, split_dir, reflected_normalB, eps)
        rabbit_regions[region].append(f)

    interaction.axes.clear()
    new_axisA = interaction.axes.add()
    new_axisA.value = reflected_normalA
    new_axisB = interaction.axes.add()
    new_axisB.value = reflected_normalB
    new_axisC = interaction.axes.add()
    new_axisC.value = axisA + axisB
    interaction.origin = origin

    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, rabbit_regions["base_A"], "base_A")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, rabbit_regions["base_B"], "base_B")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, rabbit_regions["tip_A"], "tip_A")
    new_region = interaction.regions.add()
    initialize_region_from_faces(obj, bm, new_region, rabbit_regions["tip_B"], "tip_B")

def rabbit_ear_eval(group, interaction, t, bm):

    foldA = group.folds[interaction.owner_fold_index]
    foldB = group.folds[interaction.other_fold_index]

    adjustmentAngle = interaction.adjuster

    A = fold_matrix(
        Vector(foldA.pivot_3d),
        Vector(foldA.axis).normalized(),
        foldA.angle * t
    )

    B = fold_matrix(
        Vector(foldB.pivot_3d),
        Vector(foldB.axis).normalized(),
        foldB.angle * t
    )

    angle = (foldA.angle + foldB.angle) * 0.5 * t

    C = fold_matrix(
        Vector(interaction.origin),
        Vector(interaction.axes[0].value).normalized(),
        foldB.angle * t/2
    )
    
    D = fold_matrix(
        Vector(interaction.origin),
        Vector(interaction.axes[1].value).normalized(),
        foldA.angle * t/2
    )

    E = fold_matrix(
        Vector(interaction.origin),
        Vector(interaction.axes[2].value).normalized(),
        adjustmentAngle * t
    )

    return {
        "a_base_faces": A,
        "b_base_faces": B, 
        "a_tip_faces": E @ A @ D,   
        "b_tip_faces": E @ B @ C
    }

def corner_fold_eval(group, sign, interaction, t, bm):

    angle_modifier = -0.5
    if (sign == 1):
        angle_modifier = 1.5

    foldA = group.folds[interaction.owner_fold_index]
    foldB = group.folds[interaction.other_fold_index]

    adjustmentAngle = interaction.adjuster

    upAxis = Vector(foldA.axis).cross(Vector(foldB.axis)).normalized()

    A = fold_matrix(
        Vector(foldA.pivot_3d),
        Vector(foldA.axis).normalized(),
        foldA.angle * t
    )

    B = fold_matrix(
        Vector(foldB.pivot_3d),
        Vector(foldB.axis).normalized(),
        foldB.angle * t
    )


    C = fold_matrix(
        Vector(interaction.origin),
        Vector(foldB.axis).normalized(),
        foldB.angle * t * angle_modifier
    )
    
    D = fold_matrix(
        Vector(interaction.origin),
        Vector(foldA.axis).normalized(),
        foldA.angle * t * angle_modifier
    )

    E = fold_matrix(
        Vector(interaction.origin),
        upAxis,
        adjustmentAngle * t
    )


    return {
        "a_base_faces": A,
        "b_base_faces": B, 
        "a_corner_faces": E @ A @ C,   
        "b_corner_faces": E @ B @ D
    }

def collapse_fold_eval(group, interaction, t, bm):

    foldA = group.folds[interaction.owner_fold_index]
    foldB = group.folds[interaction.other_fold_index]

    flipA = -1
    flipB = -1
    if (Vector(interaction.axes[0].value).dot(foldA.axis) > 0):
        flipA = 1
    if (Vector(interaction.axes[0].value).dot(foldB.axis) > 0):
        flipB = 1
    
    

    A = fold_matrix(
        Vector(foldA.pivot_3d),
        Vector(foldA.axis).normalized(),
        foldA.angle * t
    )

    B = fold_matrix(
        Vector(foldB.pivot_3d),
        Vector(foldB.axis).normalized(),
        foldB.angle * t
    )

    angle = (foldA.angle + foldB.angle) * 0.5 * t

    C = fold_matrix(
        Vector(interaction.origin),
        Vector(interaction.axes[0].value).normalized(),
        foldA.angle * flipA * -t
    )
    
    D = fold_matrix(
        Vector(interaction.origin),
        Vector(interaction.axes[0].value).normalized(),
        foldB.angle * flipB * -t
    )

    E = fold_matrix(
        Vector(interaction.origin),
        Vector(interaction.axes[0].value).normalized(),
        foldB.angle * flipB * t
    )

    return {
        "a_base_faces": A,
        "b_base_faces": B, 
        "a_tip_faces": A @ C,   
        "b_tip_faces": B @ D,
        "top_faces": E
    }

def nested_fold_eval(group, interaction, t, bm):

    parentFold = group.folds[interaction.owner_fold_index]
    childFold = group.folds[interaction.other_fold_index]


    A = fold_matrix(
        Vector(parentFold.pivot_3d),
        Vector(parentFold.axis).normalized(),
        parentFold.angle * t
    )

    B = fold_matrix(
        Vector(childFold.pivot_3d),
        Vector(childFold.axis).normalized(),
        childFold.angle * t
    )

    return {
        "parent_faces": A,
        "child_faces": A @ B
    }

def region_centroid(bm, face_indices):
    c = Vector((0.0, 0.0, 0.0))
    total_area = 0.0

    faces = bm.faces 

    for fi in face_indices:
        f = faces[fi]

        area = f.calc_area()
        fc = f.calc_center_median()

        c += fc * area
        total_area += area

    if total_area == 0.0:
        return Vector((0.0, 0.0, 0.0))

    return c / total_area

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n

def compute_paper_normal(bm):
    n = Vector((0, 0, 0))

    for f in bm.faces:
        n += f.normal

    if n.length < 1e-6:
        return Vector((0, 0, 1))

    return n.normalized()

def classify_face(face, base_positions, x0, split_normal):

    center = Vector((0, 0, 0))

    for v in face.verts:
        center += base_positions[v.index]

    center /= len(face.verts)

    side = (center - x0).dot(split_normal)

    if side > 1e-6:
        return 1
    elif side < -1e-6:
        return -1
    else:
        return 0
    
def build_bvh_from_bmesh(bm):
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    verts = [v.co for v in bm.verts]
    tris = []

    for f in bm.faces:
        if len(f.verts) < 3:
            continue

        v0 = f.verts[0].index
        for i in range(1, len(f.verts) - 1):
            v1 = f.verts[i].index
            v2 = f.verts[i + 1].index
            tris.append((v0, v1, v2, f.index))

    tri_indices = [(a, b, c) for (a, b, c, _) in tris]
    tri_to_face = [f_idx for (_, _, _, f_idx) in tris]

    bvh = BVHTree.FromPolygons(verts, tri_indices)

    return bvh, tri_to_face

def base_positions_from_uv(obj, bm, uv_layer):

    base_positions = [None] * len(bm.verts)

    rot = Matrix.Rotation(math.radians(obj.base_axis), 4, 'Z')

    for f in bm.faces:
        for l in f.loops:

            v = l.vert.index
            uv = l[uv_layer].uv

            p = Vector((
                (uv.x * 2) - 1,
                (uv.y * 2) - 1,
                0.0
            ))

            base_positions[v] = rot @ p

    for i, p in enumerate(base_positions):
        if p is None:
            base_positions[i] = Vector((0,0,0))

    return base_positions

def build_face_graph(bm, crease_edge_ids=None):
    if crease_edge_ids is None:
        crease_edge_ids = set()
    adjacency = {f.index: set() for f in bm.faces}

    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue

        if e.index in crease_edge_ids:
            continue

        f1, f2 = e.link_faces
        adjacency[f1.index].add(f2.index)
        adjacency[f2.index].add(f1.index)

    return adjacency

def flood_region(start_face, adjacency):
    visited = set()
    stack = [start_face]

    while stack:
        f = stack.pop()
        if f in visited:
            continue

        visited.add(f)

        for n in adjacency.get(f, []):
            if n not in visited:
                stack.append(n)

    return visited

def barycentric_coords_2d(p, a, b, c):

    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w

def uv_to_3d(bm, uv_layer, uv_point, candidate_faces=None, epsilon=1e-6):

    if candidate_faces is None:
        faces = bm.faces
    else:
        faces = [
            bm.faces[f] if isinstance(f, int) else f
            for f in candidate_faces
        ]

    for face in faces:

        loops = face.loops

        if len(loops) < 3:
            continue

        # Fan triangulation about vertex 0
        for i in range(1, len(loops) - 1):

            l0 = loops[0]
            l1 = loops[i]
            l2 = loops[i + 1]

            uv0 = Vector(l0[uv_layer].uv)
            uv1 = Vector(l1[uv_layer].uv)
            uv2 = Vector(l2[uv_layer].uv)

            bary = barycentric_coords_2d(
                uv_point,
                uv0,
                uv1,
                uv2
            )

            if bary is None:
                continue

            u, v, w = bary

            if (
                u >= -epsilon and
                v >= -epsilon and
                w >= -epsilon
            ):
                p0 = l0.vert.co
                p1 = l1.vert.co
                p2 = l2.vert.co

                return (
                    u * p0 +
                    v * p1 +
                    w * p2
                )

    return None

def fold_matrix(pivot, axis, angle):
    rot = Matrix.Rotation(angle, 4, axis)
    return Matrix.Translation(pivot) @ rot @ Matrix.Translation(-pivot)

def ensure_full_lookup_table(bm):
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    bm.faces.index_update()
    bm.edges.index_update()
    bm.verts.index_update()
    return

class FoldEvaluator:
    @staticmethod
    def evaluate(obj, timeline, preview_fold=None):
            
        me = obj.data
        
        bm = bmesh.from_edit_mesh(me) if obj.mode == 'EDIT' else bmesh.new()
        if obj.mode != 'EDIT':
            bm.from_mesh(me)

        ensure_full_lookup_table(bm)

        uv_layer = bm.loops.layers.uv.active

        uvs = [None] * len(bm.verts)

        for f in bm.faces:
            for l in f.loops:
                uv = l[uv_layer].uv
                v = l.vert.index
                uvs[v] = (uv.x, uv.y)

        raw = obj.get("origami_base_positions", None)

        if not raw or len(raw) != len(bm.verts):
            uv_layer = bm.loops.layers.uv.active
            base_positions = base_positions_from_uv(obj, bm, uv_layer)
            obj["origami_base_positions"] = [list(v) for v in base_positions]

        else:
            base_positions = [Vector(v) for v in raw]

        if preview_fold:
            group_index = int(obj.fold_timeline)

            while len(obj.fold_groups) <= group_index:
                obj.fold_groups.add()
            
        groups = obj.fold_groups

        face_xforms = {f.index: Matrix.Identity(4) for f in bm.faces}

        full = int(timeline)
        partial = timeline - full
        vert_xforms = {v.index: Matrix.Identity(4) for v in bm.verts}
        for gi, group in enumerate(groups):

            if gi >= full + 1 or (timeline == full and gi == full and not preview_fold):
                break

            group_vert_xforms = {}
            group_folds = list(group.folds)
            group_face_xforms = {f.index: [] for f in bm.faces}

            if (gi == full - 1):
                operations = build_group_operations(
                    obj,
                    bm,
                    group,
                    partial if gi == full else 1.0,
                    preview_fold,
                )
            else:
                operations = build_group_operations(
                    obj,
                    bm,
                    group,
                    partial if gi == full else 1.0,
                    None,
                )

            group_face_xforms = defaultdict(list)

            for op in operations:
                for f_idx in op.region_faces:
                    group_face_xforms[f_idx].append(op.matrix)

            for f_idx, fold_mats in group_face_xforms.items():
                if not fold_mats:
                    continue

                T = Matrix.Identity(4)

                for mat in fold_mats:
                    T = mat @ T

                face_xforms[f_idx] = T @ face_xforms[f_idx]

                f = bm.faces[f_idx]

                for v in f.verts:
                    if v.index in group_vert_xforms:
                        old = group_vert_xforms[v.index]
                        new = face_xforms[f_idx]

                    else:
                        group_vert_xforms[v.index] = face_xforms[f_idx]

            for v_idx, T in group_vert_xforms.items():
                vert_xforms[v_idx] = T

        for v_idx, T in vert_xforms.items():
            bm.verts[v_idx].co = T @ base_positions[v_idx]

        if obj.mode == 'EDIT':
            bm.normal_update()
            bmesh.update_edit_mesh(me, loop_triangles=True, destructive=False)
        else:
            bm.normal_update()
            bm.to_mesh(me)
            bm.free()

        me.update()

def draw_callback(_self):
    context = bpy.context
    obj = context.object

    if not obj or obj.name not in bpy.data.objects:
        return

    if obj.mode != 'EDIT':
        return

    preview = obj.get("_active_preview_fold")

    if not HIGHLIGHT_FACES and not preview:
        return
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    ensure_full_lookup_table(bm)

    coords = []
    mat = obj.matrix_world

    for f in bm.faces:
        if f.index not in HIGHLIGHT_FACES:
            continue

        verts = [mat @ v.co for v in f.verts]

        for i in range(1, len(verts) - 1):
            coords += [verts[0], verts[i], verts[i + 1]]

    if not coords:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": coords})

    gpu.state.blend_set('ALPHA_PREMULT')
    gpu.state.depth_test_set('NONE')

    shader.bind()
    shader.uniform_float("color", (0.0, 0.6, 0.8, 0.15))
    batch.draw(shader)

    gpu.state.blend_set('NONE')

    font_id = 0
    blf.position(font_id, 20, 40, 0)
    blf.size(font_id, 16)
    blf.draw(font_id, f"Fold Groups: {len(obj.fold_groups)}")       

class ORIGAMI_OT_pick_side(Operator):
    bl_idname = "origami.pick_side"
    bl_label = "Make New Fold"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        self.obj = context.object
        self.me = self.obj.data

        if "origami_base_positions" not in self.obj:
            bm = bmesh.from_edit_mesh(self.me)
            ensure_full_lookup_table(bm)
            print("UPDATED BASE POSITIONS")
            self.obj["origami_base_positions"] = [list(v.co) for v in bm.verts]

        self.state = 'PICK_SIDE'
        self.angle = 0.0
        self.start_mouse_x = event.mouse_region_x
        self.typing = False
        self.input_string = ""

        self.preview_fold = None

        self._init_mesh_data()

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _init_mesh_data(self):
        bm = bmesh.from_edit_mesh(self.me)

        self.edge = next((_edge for _edge in reversed(bm.select_history) if isinstance(_edge, bmesh.types.BMEdge)), None)
        if not self.edge:
            return
        
        ensure_full_lookup_table(bm)
        self.crease_edge_ids = {e.index for e in bm.edges if e.select}

        v1, v2 = self.edge.verts

        self.pivot_3d = (v1.co + v2.co) * 0.5
        self.axis = (v2.co - v1.co).normalized()

        uv_layer = bm.loops.layers.uv.active

        for loop in self.edge.link_loops:
            uv0 = loop[uv_layer].uv.copy()
            uv1 = loop.link_loop_next[uv_layer].uv.copy()

            self.crease_uv_a = uv0
            self.crease_uv_b = uv1
            break

    def modal(self, context, event):
        if self.state == 'PICK_SIDE':
            return self._modal_pick_side(context, event)

        if self.state == 'SET_ANGLE':
            return self._modal_set_angle(context, event)

        return {'RUNNING_MODAL'}

    def _modal_pick_side(self, context, event):
        if event.type == 'MOUSEMOVE':
            self._update_side_preview(context, event)

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self.locked_region = getattr(self, "current_region", set())

            self.state = 'SET_ANGLE'
            self.start_mouse_x = event.mouse_region_x
            bm = bmesh.from_edit_mesh(self.me)
            ensure_full_lookup_table(bm)
            center = Vector((0, 0, 0))
            count = 0

            for f_idx in self.current_region:
                f = bm.faces[f_idx]
                center += f.calc_center_median()
                count += 1

            if count > 0:
                center /= count
            paper_normal = compute_paper_normal(bm) 
            plane_normal = self.axis.cross(paper_normal).normalized()
            side = (center - self.pivot_3d).dot(plane_normal)

            if side < 0:
                plane_normal = -plane_normal

            self.side = side 
            self.preview_fold = PreviewFold(
                pivot_3d=self.pivot_3d,
                axis=self.axis,
                angle=0.0,
                region_faces=list(self.locked_region)
            )

            return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            HIGHLIGHT_FACES.clear()
            self.finish(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _modal_set_angle(self, context, event):

        if event.type == 'MOUSEMOVE' and not self.typing:
            delta = event.mouse_region_x - self.start_mouse_x
            self.angle = delta * 0.02

            if event.ctrl:
                step = math.radians(15)
                self.angle = round(self.angle / step) * step

        if event.value == 'PRESS':
            if event.type in {'ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE','PERIOD','MINUS'}:
                self.typing = True
                self.input_string += self._map_char(event.type)

            elif event.type == 'BACK_SPACE':
                self.input_string = self.input_string[:-1]

            elif event.type == 'RET':
                self._commit_fold(context)
                return {'FINISHED'}

        if self.typing and self.input_string:
            try:
                self.angle = math.radians(float(self.input_string))
            except:
                pass

        self.preview_fold.angle = self.angle

        timeline = self.obj.fold_timeline + 1

        FoldEvaluator.evaluate(
            self.obj,
            timeline,
            self.preview_fold
        )

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.angle = 0
            self.preview_fold.angle = self.angle
            FoldEvaluator.evaluate(
                self.obj,
                timeline,
                self.preview_fold
            )
            HIGHLIGHT_FACES.clear()
            self.finish(context)
            return {'CANCELLED'}

        if event.type == 'LEFTMOUSE':
            self._commit_fold(context)
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def _commit_fold(self, context):
        
        bm = bmesh.from_edit_mesh(self.obj.data)
        group_index = int(self.obj.fold_timeline)

        while len(self.obj.fold_groups) <= group_index:
            self.obj.fold_groups.add()

        group = self.obj.fold_groups[group_index]
        fold = group.folds.add()

        fold.pivot_3d = self.pivot_3d
        fold.axis = self.axis

        fold.uv_axis.a = self.crease_uv_a
        fold.uv_axis.b = self.crease_uv_b

        fold.angle = self.angle

        region = fold.region

        initialize_region_from_faces(
            self.obj,
            bm,
            fold.region,
            self.locked_region
        )
                    
        self.obj.fold_timeline = self.obj.fold_timeline + 1

        FoldEvaluator.evaluate(self.obj, self.obj.fold_timeline)

        bmesh.update_edit_mesh(self.me, loop_triangles=True, destructive=True)

        self.obj.update_tag()

        HIGHLIGHT_FACES.clear()

        check_new_overlapping_folds(self, context, group_index, len(group.folds)-1)

        self.finish(context)

    def _update_side_preview(self, context, event):
        region = context.region
        rv3d = context.space_data.region_3d
        coord = (event.mouse_region_x, event.mouse_region_y)

        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

        obj = self.obj
        world_to_obj = obj.matrix_world.inverted()

        origin = world_to_obj @ origin
        direction = (world_to_obj.to_3x3() @ direction).normalized()

        bm = bmesh.from_edit_mesh(self.me)
        ensure_full_lookup_table(bm)

        bvh, tri_to_face = build_bvh_from_bmesh(bm)

        hit = bvh.ray_cast(origin, direction)

        if hit[0] is None:
            return

        loc, normal, tri_index, dist = hit

        face_index = tri_to_face[tri_index]

        adjacency = build_face_graph(bm, self.crease_edge_ids)

        start_face = bm.faces[face_index]

        region = flood_region(start_face.index, adjacency)

        self.current_region = region

        HIGHLIGHT_FACES.clear()
        HIGHLIGHT_FACES.extend(region)

        context.area.tag_redraw()

    def set_timeline(obj, value):
        obj.fold_timeline = value
        FoldEvaluator.evaluate(obj, value)

    def _map_char(self, key):
        return {
            'ZERO':'0','ONE':'1','TWO':'2','THREE':'3','FOUR':'4',
            'FIVE':'5','SIX':'6','SEVEN':'7','EIGHT':'8','NINE':'9',
            'PERIOD':'.','MINUS':'-'
        }.get(key, '')

    def finish(self, context):
        if hasattr(self, "_handle"):
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')

class ORIGAMI_OT_drag_fold(Operator):
    bl_idname = "origami.drag_fold"
    bl_label = "Drag Corner Fold"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        self.obj = context.object
        self.me = self.obj.data

        self.snap_enabled = False
        self._cached_raycast = None
        self._cached_plane = None
        self._cached_path = None
        self.preview_fold = None
        self.start_vertex = None
        self.side_sign = None
        self.crease_path = None

        bm = bmesh.from_edit_mesh(self.me)
        ensure_full_lookup_table(bm)

        self.kd = KDTree(len(bm.verts))
        for v in bm.verts:
            self.kd.insert(v.co, v.index)
        self.kd.balance()

        v = next((v for v in bm.verts if v.select), None)
        if not v:
            self.report({'WARNING'}, "Select a vertex")
            return {'CANCELLED'}

        self.start_vertex = v.co.copy()
        self.start_vertex_index = v.index

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_preview, (context,), 'WINDOW', 'POST_VIEW'
        )
        DRAW_HANDLERS.add(self._handle)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            HIGHLIGHT_FACES.clear()
            self.finish(context)
            return {'CANCELLED'}

        self.snap_enabled = (
            event.shift or
            event.ctrl
        )
            
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self.update_preview(context, event)

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self.commit(context)
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def raycast_point(self, context, event):
        bm = bmesh.from_edit_mesh(self.me)
        ensure_full_lookup_table(bm)
        region = context.region
        rv3d = context.space_data.region_3d
        coord = (event.mouse_region_x, event.mouse_region_y)

        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

        depsgraph = context.evaluated_depsgraph_get()

        hit, loc, normal, face_index, obj, _ = context.scene.ray_cast(
            depsgraph, origin, direction
        )
        if not hit:
            return None
        obj = self.obj
        world_to_obj = obj.matrix_world.inverted()

        B_world = loc
        B_obj = world_to_obj @ B_world

        if self.snap_enabled:
            B_obj, snap_type = self.get_snapped_point(bm, B_obj)
        else:
            snap_type = "NONE"

        self.snap_type = snap_type

        return B_obj if hit else None

    def update_preview(self, context, event):

        B = self.raycast_point(context, event)
        if not B:
            return
        self.drag_point = B.copy()

        bm = bmesh.from_edit_mesh(self.me)
        self._faces_snapshot = [f.index for f in bm.faces]
        ensure_full_lookup_table(bm)

        start_vert = bm.verts[self.start_vertex_index]

        if not start_vert:
            return

        if not start_vert.link_faces:
            return

        seed_face = start_vert.link_faces[0]

        candidate_faces = collect_sheet_region(
            bm,
            seed_face
        )

        n = Vector()

        for f in candidate_faces:
            n += f.normal

        normal = n.normalized()

        pivot, axis = self.crease_from_points(
            self.start_vertex,
            B,
            normal
        )

        plane_normal = axis.cross(normal).normalized()

        self._cached_plane = (pivot, plane_normal)

        self.preview_region = []

        for f in candidate_faces:

            center = f.calc_center_median()

            d = (center - pivot).dot(plane_normal)

            if d > 0:
                self.preview_region.append(f.index)
                
        threshold = 0.005 * self.obj.scale.length

        self._cached_path = build_crease_path(
            bm,
            candidate_faces,
            pivot,
            plane_normal,
            threshold
        )
        self._cached_bm = bm

        self.crease_path = self._cached_path

        context.area.tag_redraw()

    def draw_preview(self, context):
        
        if not self.crease_path:
            return

        obj = self.obj
        obj_to_world = obj.matrix_world

        bm = bmesh.from_edit_mesh(self.me)
        ensure_full_lookup_table(bm)     

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        new_points = []
        existing_points = []

        new_lines = []
        existing_lines = []

        if len(self.crease_path.nodes) >= 2:

            plane_origin, plane_normal = self._cached_plane

            nodes = self.crease_path.nodes

            if len(nodes) < 2:
                return

            max_dist = -1.0
            first_node = None
            last_node = None

            for i in range(len(nodes)):

                for j in range(i + 1, len(nodes)):

                    a = nodes[i]
                    b = nodes[j]

                    d = (a.co - b.co).length_squared

                    if d > max_dist:

                        max_dist = d
                        first_node = a
                        last_node = b

            if first_node is None or last_node is None:
                return

            p0 = first_node.co
            p1 = last_node.co

            tri = [
                obj_to_world @ p0,
                obj_to_world @ p1,
                obj_to_world @ self.drag_point
            ]

            gpu.state.blend_set('ALPHA')

            batch = batch_for_shader(
                shader,
                'TRIS',
                {"pos": tri}
            )

            shader.bind()

            shader.uniform_float(
                "color",
                (0.2, 0.7, 1.0, 0.25)
            )

            batch.draw(shader)

            gpu.state.blend_set('NONE')
            
        for node in self.crease_path.nodes:

            world = obj_to_world @ node.co

            if node.kind == "NEW_VERT":
                new_points.append(world)
            else:
                existing_points.append(world)

        for seg in self.crease_path.segments:

            a = obj_to_world @ seg.a.co
            b = obj_to_world @ seg.b.co

            if seg.is_existing:
                existing_lines += [a, b]
            else:
                new_lines += [a, b]

        if existing_lines:

            batch = batch_for_shader(
                shader,
                'LINES',
                {"pos": existing_lines}
            )

            shader.bind()
            shader.uniform_float("color", (0.2, 1.0, 1.0, 1.0))

            gpu.state.line_width_set(3.0)

            batch.draw(shader)

        if new_lines:

            batch = batch_for_shader(
                shader,
                'LINES',
                {"pos": new_lines}
            )

            shader.bind()
            shader.uniform_float("color", (1.0, 0.5, 0.0, 1.0))

            gpu.state.line_width_set(3.0)

            batch.draw(shader)

        if existing_points:

            batch = batch_for_shader(
                shader,
                'POINTS',
                {"pos": existing_points}
            )

            shader.bind()
            shader.uniform_float("color", (0.0, 1.0, 0.0, 1.0))

            gpu.state.point_size_set(8)

            batch.draw(shader)

        if new_points:

            batch = batch_for_shader(
                shader,
                'POINTS',
                {"pos": new_points}
            )

            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))

            gpu.state.point_size_set(8)

            batch.draw(shader)

    def snap_to_vertex(self, p_obj, threshold):
        co, index, dist = self.kd.find(p_obj)
        if dist < threshold:
            return co, "VERT"
        return None, None

    def snap_to_edge(self, bm, p_obj, threshold):
        best_p = None
        best_dist = threshold

        for e in bm.edges:
            a = e.verts[0].co
            b = e.verts[1].co

            p = closest_point_on_edge(p_obj, a, b)
            d = (p - p_obj).length

            if d < best_dist:
                best_dist = d
                best_p = p

        if best_p:
            return best_p, "EDGE"

        return None, None

    def get_snapped_point(self, bm, p_obj):
        threshold = 0.02 * self.obj.scale.length

        v_snap, v_type = self.snap_to_vertex(p_obj, threshold)
        if v_snap:
            return v_snap, v_type

        e_snap, e_type = self.snap_to_edge(bm, p_obj, threshold)
        if e_snap:
            return e_snap, e_type

        return p_obj, "NONE"

    def commit(self, context):
        if not self.crease_path:
            print("NO VALID CREASE PATH")
            self.finish(context)
            return
        bm = bmesh.from_edit_mesh(self.me)
        ensure_full_lookup_table(bm)

        threshold = 0.01 * self.obj.scale.length
        
        new_verts = apply_crease_path(
            self._cached_bm,
            self._cached_path,
            threshold
        )

        self._cached_bm.select_flush_mode()

        self._cached_bm.select_history.clear()

        for v in self._cached_bm.verts:
            v.select = False

        for v in new_verts:
            if v.is_valid:
                v.select = True

        self._cached_bm.select_flush(True)
        bmesh.update_edit_mesh(self.me, loop_triangles=True, destructive=True)

        HIGHLIGHT_FACES.clear()
        self.finish(context)

    def compute_paper_normal(self, bm):
        n = Vector()
        for f in bm.faces:
            n += f.normal
        return n.normalized()

    def crease_from_points(self, A, B, normal):
        mid = (A + B) * 0.5
        ab = (B - A).normalized()
        axis = normal.cross(ab).normalized()
        return mid, axis

    def classify_faces(self, bm, pivot, axis, normal):
        return set(f.index for f in bm.faces), axis.cross(normal).normalized()

    def finish(self, context):
        if hasattr(self, "_handle"):
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')

class ORIGAMI_OT_update_fold_axis(Operator):
    bl_idname = "origami.update_fold_axis"
    bl_label = "Update Fold Axis"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        
        obj = context.object

        bm = bmesh.from_edit_mesh(obj.data)

        self.edge = next((_edge for _edge in reversed(bm.select_history) if isinstance(_edge, bmesh.types.BMEdge)), None)
        if not self.edge:
            return
        
        ensure_full_lookup_table(bm)
        self.crease_edge_ids = {e.index for e in bm.edges if e.select}

        v1, v2 = self.edge.verts

        self.pivot_3d = (v1.co + v2.co) * 0.5
        self.axis = (v2.co - v1.co).normalized()

        group = obj.fold_groups[obj.active_fold_group]
        fold = group.folds[obj.active_fold]

        fold.pivot_3d = self.pivot_3d

        uv_layer = bm.loops.layers.uv.active

        loop = self.edge.link_loops[0]

        uv0 = loop[uv_layer].uv
        uv1 = loop.link_loop_next[uv_layer].uv

        if Vector(fold.axis).dot(Vector(self.axis)) > 0:

            fold.axis = self.axis

            fold.uv_axis.a = uv0
            fold.uv_axis.b = uv1
        else:
            fold.axis = self.axis * -1

            fold.uv_axis.a = uv1
            fold.uv_axis.b = uv0


        return {'FINISHED'}

class ORIGAMI_OT_update_fold_axes_from_uv(Operator):
    bl_idname = "origami.update_fold_axes_from_uv"
    bl_label = "Update Fold Axes From UV"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):

        obj = context.object
        me = obj.data

        bm = bmesh.from_edit_mesh(me)

        ensure_full_lookup_table(bm)

        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            self.report({'ERROR'}, "Mesh has no UV map.")
            return {'CANCELLED'}

        for gid, group in enumerate(obj.fold_groups):
            FoldEvaluator.evaluate(obj, gid)
            for fold in group.folds:

                segment = fold.uv_axis

                p0 = uv_to_3d(
                    bm,
                    uv_layer,
                    Vector(segment.a)
                )

                p1 = uv_to_3d(
                    bm,
                    uv_layer,
                    Vector(segment.b)
                )

                if p0 is None or p1 is None:
                    self.report(
                        {'WARNING'},
                        f"Could not reconstruct fold '{group.name}'."
                    )
                    continue

                new_axis = (p1 - p0)
                
                if Vector(fold.axis).dot(new_axis) > 0:
                    fold.axis = new_axis
                else:
                    fold.axis = new_axis * -1

                fold.pivot_3d = (p0 + p1) * 0.5

        bmesh.update_edit_mesh(me, loop_triangles=False)

        self.report({'INFO'}, "Updated fold axes from UV.")
        return {'FINISHED'}
    
class ORIGAMI_OT_set_basis(bpy.types.Operator):
    bl_idname = "origami.set_basis"
    bl_label = "Set Origami Basis"

    def execute(self, context):

        obj = context.object
        bm = bmesh.from_edit_mesh(obj.data)

        ensure_full_lookup_table(bm)

        f = bm.faces[0]

        v0 = f.verts[0].co
        v1 = f.verts[1].co
        v3 = f.verts[-1].co

        origin = f.calc_center_median()

        x_axis = (v1 - v0)
        y_axis = (v3 - v0)

        obj["origami_basis_origin"] = list(origin)
        obj["origami_basis_x"] = list(x_axis)
        obj["origami_basis_y"] = list(y_axis)

        self.report({'INFO'}, "Origami basis stored")
        uv_layer = bm.loops.layers.uv.active
        base_positions = base_positions_from_uv(obj, bm, uv_layer)
        obj["origami_base_positions"] = [list(v) for v in base_positions]
        FoldEvaluator.evaluate(obj, 0)
        return {'FINISHED'}

class ORIGAMI_OT_fold_back(bpy.types.Operator):
    bl_idname = "origami.fold_back"
    bl_label = "Previous Fold"

    def execute(self, context):
        obj = context.object

        obj.fold_timeline = max(0.0, math.floor(obj.fold_timeline-0.0001))

        FoldEvaluator.evaluate(obj, obj.fold_timeline)

        return {'FINISHED'}

class ORIGAMI_OT_fold_forward(bpy.types.Operator):
    bl_idname = "origami.fold_forward"
    bl_label = "Next Fold"

    def execute(self, context):
        obj = context.object

        obj.fold_timeline = min(len(obj.fold_groups), math.ceil(obj.fold_timeline + 0.0001))

        FoldEvaluator.evaluate(obj, obj.fold_timeline)

        return {'FINISHED'}

class ORIGAMI_OT_fold_latest(bpy.types.Operator):
    bl_idname = "origami.fold_latest"
    bl_label = "Latest Fold"

    def execute(self, context):
        obj = context.object

        obj.fold_timeline = float(len(obj.fold_groups))

        FoldEvaluator.evaluate(obj, obj.fold_timeline)

        return {'FINISHED'}

class ORIGAMI_PT_panel(Panel):
    bl_label = "Origami Fold"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Origami"

    def draw(self, context):
        layout = self.layout
        obj = context.object

        layout.operator("origami.pick_side", text="Make New Fold")
        layout.operator("origami.drag_fold", text="Drag Corner Fold")
        layout.operator("origami.update_fold_axis", text="Update Fold Axis")
        layout.operator("origami.update_fold_axes_from_uv", text="Update All Axes From UV")
        layout.operator("origami.fix_stuff", text="Fix Broken Stuff")
        layout.prop(obj, "base_axis", slider=True)
        layout.operator("origami.set_basis", text="Set New Base Axis Rotation")

class ORIGAMI_OT_fix_stuff(bpy.types.Operator):
    bl_idname = "origami.fix_stuff"
    bl_label = "Fix Everything"
    
    def execute(self, context):
        obj = context.object
        bm = bmesh.from_edit_mesh(obj.data)
        rebuild_regions_from_seed_uv(obj, bm)
        HIGHLIGHT_FACES.clear()
        clean_up_interactions(self, context)
        return{'FINISHED'}

class ORIGAMI_OT_fold_delete(bpy.types.Operator):
    bl_idname = "origami.fold_delete"
    bl_label = "Delete Fold"

    def execute(self, context):
        obj = context.object
        groups = obj.fold_groups
        g = obj.active_fold_group
        f = obj.active_fold

        group = obj.fold_groups[g]

        if f < len(group.folds):
            fold = group.folds[f]
            for other_fold in group.folds:
                for i in range(len(other_fold.interactions) - 1, -1, -1):
                    interaction = other_fold.interactions[i]

                    if interaction.other_fold_index == f:
                        other_fold.interactions.remove(i)

                    elif interaction.other_fold_index > f:
                        interaction.other_fold_index -= 1

            group.folds.remove(f)
            obj.active_fold = max(0, f - 1)
        if len(group.folds) == 0:
            groups.remove(g)
        if len(groups) > 0 and obj.active_fold_group == len(groups):
            obj.active_fold_group = obj.active_fold_group - 1
        
        return {'FINISHED'}

def copy_region(src_region, dst_region):
    dst_region.region_name = src_region.region_name
    dst_region.seed_uv = src_region.seed_uv

    for face in src_region.cached_faces:
        item = dst_region.cached_faces.add()
        item.index = face.index

    for seg in src_region.crease_uv_segments:
        item = dst_region.crease_uv_segments.add()
        item.a = seg.a
        item.b = seg.b

class ORIGAMI_OT_fold_move_up(bpy.types.Operator):
    bl_idname = "origami.fold_move_up"
    bl_label = "Move Fold Up"

    def execute(self, context):
        obj = context.object

        g = obj.active_fold_group
        f = obj.active_fold

        group = obj.fold_groups[g]

        if f > 0:
            group.folds.move(f, f - 1)
            obj.active_fold -= 1
            return {'FINISHED'}

        if g <= 0:
            return {'CANCELLED'}

        src_group = group
        dst_group = obj.fold_groups[g - 1]

        if f >= len(src_group.folds):
            return {'CANCELLED'}

        fold = src_group.folds[f]

        new_fold = dst_group.folds.add()

        for attr in ["pivot_3d", "axis", "angle"]:
            setattr(new_fold, attr, getattr(fold, attr))

        copy_region(
            fold.region,
            new_fold.region
        )

        src_group.folds.remove(f)

        obj.active_fold_group = g - 1
        obj.active_fold = len(dst_group.folds) - 1
        clean_up_interactions(self, context)

        check_new_overlapping_folds(self, context, g - 1, len(dst_group.folds)-1)

        return {'FINISHED'}

class ORIGAMI_OT_fold_move_down(bpy.types.Operator):
    bl_idname = "origami.fold_move_down"
    bl_label = "Move Fold Down"

    def execute(self, context):
        obj = context.object

        g = obj.active_fold_group
        f = obj.active_fold

        group = obj.fold_groups[g]

        if f < len(group.folds) - 1:
            group.folds.move(f, f + 1)
            obj.active_fold += 1
            return {'FINISHED'}

        if g >= len(obj.fold_groups) - 1:
            obj.fold_groups.add()

        src_group = group
        dst_group = obj.fold_groups[g + 1]

        if f >= len(src_group.folds):
            return {'CANCELLED'}

        fold = src_group.folds[f]

        new_fold = dst_group.folds.add()

        for attr in ["pivot_3d", "axis", "angle"]:
            setattr(new_fold, attr, getattr(fold, attr))

        copy_region(
            fold.region,
            new_fold.region
        )

        src_group.folds.remove(f)

        obj.active_fold_group = g + 1
        obj.active_fold = len(dst_group.folds) - 1
        clean_up_interactions(self, context)

        check_new_overlapping_folds(self, context, g + 1, len(dst_group.folds)-1)

        return {'FINISHED'}

class ORIGAMI_OT_group_move_up(bpy.types.Operator):
    bl_idname = "origami.group_move_up"
    bl_label = "Move Group Up"

    def execute(self, context):
        obj = context.object
        i = obj.active_fold_group

        if i > 0:
            obj.fold_groups.move(i, i - 1)
            obj.active_fold_group -= 1

        return {'FINISHED'}

class ORIGAMI_OT_group_move_down(bpy.types.Operator):
    bl_idname = "origami.group_move_down"
    bl_label = "Move Group Down"

    def execute(self, context):
        obj = context.object
        i = obj.active_fold_group

        if i < len(obj.fold_groups):
            obj.fold_groups.move(i, i + 1)
            obj.active_fold_group += 1

        return {'FINISHED'}

class ORIGAMI_OT_flip_axis(Operator):
    bl_idname = "origami.flip_fold_axis"
    bl_label = "Flip the Axis of a Fold"

    def execute(self, context):
        obj = context.object
        bm = bmesh.from_edit_mesh(obj.data)
        ensure_full_lookup_table(bm)
        group = obj.fold_groups[obj.active_fold_group]

        for fold in group.folds:
            if fold.selected:
                fold.axis = (
                    -fold.axis[0],
                    -fold.axis[1],
                    -fold.axis[2]
                )
                fold.angle = -fold.angle

        return {'FINISHED'}

class ORIGAMI_PT_fold_history_panel(Panel):
    bl_label = "Fold History"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Origami"

    def draw(self, context):
        layout = self.layout
        obj = context.object

        if not obj:
            return

        layout.label(text="Timeline Controls")
        layout.prop(obj, "fold_timeline", slider=True)
        if(len(obj.fold_groups) > 0):
            row = layout.row()
            row.operator("origami.fold_back", text="◀ Back")
            row.operator("origami.fold_forward", text="Forward ▶")
            layout.operator("origami.fold_latest", text="⏩ Latest")
            if obj.active_fold_group < len(obj.fold_groups):
                layout.operator("origami.flip_fold_axis", text=f"Flip Selected Folds Axis")
            
            row = layout.row()
            row.template_list(
                "ORIGAMI_UL_fold_groups",
                "",
                obj,
                "fold_groups",
                obj,
                "active_fold_group",
                rows=4
            )

            col = row.column(align=True)
            col.operator("origami.group_move_up", icon='TRIA_UP', text="")
            col.operator("origami.group_move_down", icon='TRIA_DOWN', text="")
            group = obj.fold_groups[obj.active_fold_group]
            if obj.fold_groups and obj.active_fold_group < len(obj.fold_groups):
                layout.label(text="Folds")
                row = layout.row()
                row.template_list(
                    "ORIGAMI_UL_folds",
                    "",
                    group,
                    "folds",
                    obj,
                    "active_fold",
                    rows=6
                )

                col = row.column(align=True)
                col.operator("origami.fold_delete", icon='X', text="")
                col.operator("origami.fold_move_up", icon='TRIA_UP', text="")
                col.operator("origami.fold_move_down", icon='TRIA_DOWN', text="")

            if (group.folds and obj.active_fold < len(group.folds)):
                active_fold = group.folds[obj.active_fold]
                box = layout.box()
                box.label(text="Fold Interactions")
                interactions = interactions_for_fold(group, obj.active_fold)
                for interaction in interactions:

                    row = box.row()

                    row.label(
                        text=f"Fold {interaction.owner_fold_index if interaction.other_fold_index == obj.active_fold else interaction.other_fold_index}"
                    )
                    row.prop(
                        interaction,
                        "adjuster",
                        text=""
                    )
                    row.prop(
                        interaction,
                        "interaction_type",
                        text=""
                    )

def register():
    enable_uv_debug()
    bpy.utils.register_class(OrigamiFaceIndex)
    bpy.utils.register_class(OrigamiUVSegment)
    bpy.utils.register_class(OrigamiAxis)
    bpy.utils.register_class(OrigamiRegion)
    bpy.utils.register_class(FoldInteraction)
    bpy.utils.register_class(OrigamiFold)
    bpy.utils.register_class(RabbitEar)
    bpy.utils.register_class(CornerFold)
    bpy.utils.register_class(OrigamiFoldGroup)
    bpy.utils.register_class(ORIGAMI_OT_pick_side)
    bpy.utils.register_class(ORIGAMI_OT_drag_fold)
    bpy.utils.register_class(ORIGAMI_OT_update_fold_axis)
    bpy.utils.register_class(ORIGAMI_OT_update_fold_axes_from_uv)
    bpy.utils.register_class(ORIGAMI_OT_set_basis)
    bpy.utils.register_class(ORIGAMI_PT_panel)
    bpy.utils.register_class(ORIGAMI_PT_fold_history_panel)

    bpy.utils.register_class(ORIGAMI_OT_fix_stuff)
    bpy.utils.register_class(ORIGAMI_UL_fold_groups)
    bpy.utils.register_class(ORIGAMI_UL_folds)
    bpy.utils.register_class(ORIGAMI_UL_fold_interaction)
    bpy.utils.register_class(ORIGAMI_OT_fold_delete)
    bpy.utils.register_class(ORIGAMI_OT_fold_move_up)
    bpy.utils.register_class(ORIGAMI_OT_fold_move_down)
    bpy.utils.register_class(ORIGAMI_OT_group_move_up)
    bpy.utils.register_class(ORIGAMI_OT_group_move_down)
    bpy.utils.register_class(ORIGAMI_OT_flip_axis)

    bpy.utils.register_class(ORIGAMI_OT_fold_back)
    bpy.utils.register_class(ORIGAMI_OT_fold_forward)
    bpy.utils.register_class(ORIGAMI_OT_fold_latest)

    bpy.types.Object.fold_groups = CollectionProperty(type=OrigamiFoldGroup)
    bpy.types.Object.base_axis = FloatProperty(
        default=0.0,
        min=0.0,
        max=90.0,
        soft_max=90,
    )
    bpy.types.Object.fold_timeline = FloatProperty(
        default=0.0,
        min=0.0,
        max=1000.0,
        soft_max=10,
        update=update_timeline
    )
    bpy.types.Object.origami_paper_positions = CollectionProperty(
        type=bpy.types.PropertyGroup
    )
    bpy.types.Object.active_fold_group = IntProperty(
        update=on_active_group_changed
    )
    bpy.types.Object.active_fold = IntProperty(
        update=on_active_fold_changed
    )
    bpy.types.Object.active_rabbit = IntProperty()

    if depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_handler)
    if origami_frame_update not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(origami_frame_update)
    if undo_post_handler not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(undo_post_handler)

    ensure_draw_handler()

def ensure_draw_handler():
    if not DRAW_HANDLERS:
        handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback, (None,), 'WINDOW', 'POST_VIEW'
        )
        DRAW_HANDLERS.add(handle)

def unregister():
    disable_uv_debug()
    bpy.utils.unregister_class(OrigamiFaceIndex)
    bpy.utils.unregister_class(OrigamiUVSegment)
    bpy.utils.unregister_class(OrigamiAxis)
    bpy.utils.unregister_class(OrigamiRegion)
    bpy.utils.unregister_class(FoldInteraction)
    bpy.utils.unregister_class(ORIGAMI_PT_fold_history_panel)

    bpy.utils.unregister_class(ORIGAMI_OT_fix_stuff)
    bpy.utils.unregister_class(ORIGAMI_UL_fold_groups)
    bpy.utils.unregister_class(ORIGAMI_UL_folds)
    bpy.utils.unregister_class(ORIGAMI_UL_fold_interaction)
    bpy.utils.unregister_class(ORIGAMI_OT_fold_delete)
    bpy.utils.unregister_class(ORIGAMI_OT_fold_move_up)
    bpy.utils.unregister_class(ORIGAMI_OT_fold_move_down)
    bpy.utils.unregister_class(ORIGAMI_OT_group_move_up)
    bpy.utils.unregister_class(ORIGAMI_OT_group_move_down)
    bpy.utils.unregister_class(ORIGAMI_OT_flip_axis)

    bpy.utils.unregister_class(ORIGAMI_PT_panel)
    bpy.utils.unregister_class(ORIGAMI_OT_pick_side)
    bpy.utils.unregister_class(ORIGAMI_OT_drag_fold)
    bpy.utils.unregister_class(ORIGAMI_OT_update_fold_axis)
    bpy.utils.unregister_class(ORIGAMI_OT_update_fold_axes_from_uv)
    bpy.utils.unregister_class(ORIGAMI_OT_set_basis)
    bpy.utils.unregister_class(OrigamiFold)
    bpy.utils.unregister_class(OrigamiFoldGroup)
    bpy.utils.unregister_class(RabbitEar)
    bpy.utils.unregister_class(CornerFold)
    bpy.utils.unregister_class(ORIGAMI_OT_fold_back)
    bpy.utils.unregister_class(ORIGAMI_OT_fold_forward)
    bpy.utils.unregister_class(ORIGAMI_OT_fold_latest)

    del bpy.types.Object.fold_groups
    del bpy.types.Object.fold_timeline
    del bpy.types.Object.base_axis
    del bpy.types.Object.origami_paper_positions
    del bpy.types.Object.active_fold_group
    del bpy.types.Object.active_fold
    del bpy.types.Object.active_rabbit

    remove_draw_handlers()
    if depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_handler)
    if origami_frame_update in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(origami_frame_update)
    if undo_post_handler in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(undo_post_handler)

def remove_draw_handlers():
    for h in DRAW_HANDLERS:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
        except:
            pass
    DRAW_HANDLERS.clear()
    print("Removed draw handlers")

remove_draw_handlers()

if __name__ == "__main__":
    register()