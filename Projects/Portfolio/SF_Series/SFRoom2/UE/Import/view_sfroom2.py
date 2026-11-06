"""Point the level viewport at the room (same view as the Blender viewport) and save a screenshot.

    py "E:/GitHub/Graphics/Projects/Portfolio/SFRoom2/UE/Import/view_sfroom2.py"

The image lands in Saved/Screenshots/WindowsEditor/.
"""
import unreal

CAMERA_LOC = unreal.Vector(149.08, 186.60, 172.89)
CAMERA_FWD = unreal.Vector(0.2755, -0.9609, -0.0279)

unreal.get_editor_subsystem(unreal.EditorActorSubsystem).select_nothing()
unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
    CAMERA_LOC, unreal.MathLibrary.make_rot_from_x(CAMERA_FWD))
unreal.SystemLibrary.execute_console_command(None, "HighResShot 1600x900")
