# blender-origami
A project I'm working on in Blender to allow you to fold and crease a flat mesh just like you would a piece of paper. Currently it's designed to export to Alembic.
If anything is ever looking really broken, go to edit mode on the object and hit "fix broken stuff" as a first line of defence.

--How to use it--

1) How to start:
    Always start with a new basic plane. Currently this add-on only supports the basic square plane. You can scale or move it in object mode but don't try to move vert in edit mode except with the origami tool.
    Once you have the add-on installed the "Origami" tab will appear on the sidebar on the right. Click it to view the operators and fold history panel.

2) Making Folds:
    In order to create new folds, you need edges on your mesh that create full crease lines. Select the edges you want to be your crease and then click "Make New Fold."
    Then hover your mouse over whichever side of the crease you want to move. This will highlight all faces in that region and set your fold axis based on your currently selected edge.
    Finally, move your mouse across the screen to set your desired fold angle and click to confirm. You will then see your fold appear in the current group in the "Fold History" window.
    If you are adding a fold at the end of the timeline, a new group will be created. After adding a fold, the fold_timeline will always be incremented by 1.
    
3) Modifying Folds:
    The angle of folds can be modified in the Folds section of the Fold History window. 
    If the axis of a fold needs to be updated, select that fold in the window, select the edge that should define the new axis, and click "Update Fold Axis.
    Fold axes are not attached to edges as this system allows for arbitrary topology changes which could invalidate edge tracking. Fold axes are stored in 3d object space.
    If you have modified earlier fold angles in such a way that would misalign future fold axes, press "Update All Axes From UV" to update axes based on the UV crease graph.

4) Fold Interactions:
    If two folds ever have overlapping regions, a fold interaction will automatically be generated. It can be seen at the bottom of the Fold History tab when a participating fold is selected.
    Fold Interactions have a default type of None. When this type is selected, rotation matrices are applied as fold1Matrix @ fold0Matrix. So lower folds happen in the relative space of higher folds.
    Other fold types are some common folds that occur when you try to fold two overlapping parts of the paper at once. Setting these interaction types will produce (mostyly) physically realistic behavior.

5) Fold and Group Management:
    Folds and Groups can be moved up and down the timeline using the arrow buttons on the side of their display areas. Folds can also be deleted but groups are only deleted when all folds within them are deleted.
    Empty groups are possible only if a fold is created and then moved out of its group. An empty group can be deleted if the fold delete button is clicked while selecting an empty group.
    Whichever group a fold is in determines when it happens in the timeline. Group 0 begins its evaluation at timeline=0 and ends at timeline=1. Group 5 begins evaluation at timeline=0 and ends at timeline=6.

6) Animation and Export:
    In order to animate the origami folding or unfolding, you just have to set keyframes for the fold_timeline. You can do this by hovering over the timeline and pressing "i" on your keyboard.
    The object position, rotation, and scale(sometimes) in Object mode can all be modified without disrupting the  folding animation.
    This only possible export format is an Alembic (.abc) file. This is because the folding script that reads the data only exists within Blender. But .abc saves mesh data per frame so it preserves the folding.
    One could theoretically recreate the script that reads the data in another program but for now this is the simplest solution.

    CRUCIAL EXPORT NOTE:
    Currently Alembic export won't work unless there is some kind of modifier on the blender object during export. Any modifier will do. Tried and failed to fix this many times.
    If you don't have a modifier and don't want one you can add a Weld modifier with distance=0 which will essentially just do nothing but allow the export to work properly.

--Important note about UVs--
    It took me a long time to figure out the best appraoch for this. I thought a skeletal rig or shape keys might work for a time but I kept running into issues with those. 
    One major problem I faced with this particular implementation was how to make it robust against topology changes. I was using edge and face IDs for awhile but as soon as I cut the mesh it broke.
    I needed a way to add new crease lines while building without invalidating all folds. The solution I ended up with was to use the flat piece of paper as the source of truth using UVs.
    All of the definitions are stored in UV space so if you mess around with the UV mapping it will break. But it allows the system to know exactly what regions are what without using face or edge IDs.