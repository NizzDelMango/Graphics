"""Import the SFRoom2 Blender export (FBX per object + layout.json) into the open Unreal level.

Run from the Unreal Output Log (Cmd -> Python):
    py "E:/GitHub/Graphics/Projects/Portfolio/SFRoom2/UE/Import/import_sfroom2.py"

Re-running replaces the assets and the actors in the "SFRoom2" outliner folder.
"""
import json
import math
import os

import unreal

IMPORT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_ROOT = "/Game/SFRoom2/Meshes"
OUTLINER_FOLDER = "SFRoom2"

actor_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

with open(os.path.join(IMPORT_DIR, "layout.json")) as f:
    layout = json.load(f)


def import_fbx(name):
    task = unreal.AssetImportTask()
    task.filename = os.path.join(IMPORT_DIR, name + ".fbx")
    task.destination_path = CONTENT_ROOT + "/" + name
    task.automated = True
    task.replace_existing = True
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    for path in task.imported_object_paths:
        asset = unreal.load_asset(path)
        if isinstance(asset, unreal.StaticMesh):
            return asset
    return None


def to_rotator(quat_xyzw):
    x, y, z, w = quat_xyzw
    return unreal.Quat(x, y, z, w).rotator()


# Clear actors from a previous run.
for actor in actor_sub.get_all_level_actors():
    if str(actor.get_folder_path()).startswith(OUTLINER_FOLDER):
        actor_sub.destroy_actor(actor)

missing = []
with unreal.ScopedSlowTask(len(layout["meshes"]), "Importing SFRoom2") as slow_task:
    slow_task.make_dialog(True)
    for item in layout["meshes"]:
        if slow_task.should_cancel():
            break
        slow_task.enter_progress_frame(1, item["name"])
        mesh = import_fbx(item["name"])
        if mesh is None:
            missing.append(item["name"])
            continue
        actor = actor_sub.spawn_actor_from_object(
            mesh, unreal.Vector(*item["loc"]), to_rotator(item["quat_xyzw"]))
        actor.set_actor_scale3d(unreal.Vector(*item["scale"]))
        actor.set_actor_label(item["name"])
        actor.set_folder_path(OUTLINER_FOLDER + "/Meshes")

# Blender watts -> Unreal candela (683 lm/W): point = P/4pi, area (one-sided) = P/pi.
for item in layout["lights"]:
    loc = unreal.Vector(*item["loc"])
    if item["type"] == "AREA":
        actor = actor_sub.spawn_actor_from_class(unreal.RectLight, loc)
        comp = actor.light_component
        size_cm = (item["size_m"] or 1.0) * 100
        comp.set_editor_property("source_width", size_cm)
        comp.set_editor_property("source_height", size_cm)
        candela = item["energy_w"] * 683 / math.pi
        # Blender area lights shine down their local -Z; Unreal rect lights shine along +X.
        q = unreal.Quat(*item["quat_xyzw"])
        actor.set_actor_rotation(unreal.MathLibrary.make_rot_from_x(q.rotate_vector(unreal.Vector(0, 0, -1))), False)
    else:
        actor = actor_sub.spawn_actor_from_class(unreal.PointLight, loc)
        comp = actor.light_component
        candela = item["energy_w"] * 683 / (4 * math.pi)
    comp.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
    comp.set_editor_property("intensity", candela)
    comp.set_light_color(unreal.LinearColor(*item["color"], 1.0))
    comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    actor.set_actor_label(item["name"])
    actor.set_folder_path(OUTLINER_FOLDER + "/Lights")

unreal.log("SFRoom2 import done: {} meshes, {} lights, missing: {}".format(
    len(layout["meshes"]) - len(missing), len(layout["lights"]), missing))
