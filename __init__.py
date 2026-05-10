bl_info = {
    "name": "Origami Addon",
    "author": "You",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "category": "Object",
}

import importlib

# --- Import modules ---
if "bpy" in locals():
    importlib.reload(origami_main)
    importlib.reload(origami_crease_visualizer)
    importlib.reload(origami_state)
else:
    from . import origami_main
    from . import origami_crease_visualizer
    from . import origami_state

# List of modules for easy register/unregister
modules = [
    origami_main,
    origami_crease_visualizer,
    origami_state,
]


# --- Register ---
def register():
    for m in modules:
        if hasattr(m, "register"):
            m.register()


def unregister():
    for m in reversed(modules):
        if hasattr(m, "unregister"):
            try:
                m.unregister()
            except Exception as e:
                print(f"Unregister failed: {e}")
