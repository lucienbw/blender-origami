bl_info = {
    "name": "Blender-Origami",
    "author": "You",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "category": "Object",
}

import importlib

# --- Import modules ---
if "bpy" in locals():
    importlib.reload(origami_state)
    importlib.reload(origami_crease_visualizer)
    importlib.reload(origami_main)
else:
    from . import origami_state
    from . import origami_crease_visualizer
    from . import origami_main
    


# List of modules for easy register/unregister
modules = [
    origami_state,
    origami_crease_visualizer,
    origami_main,
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
