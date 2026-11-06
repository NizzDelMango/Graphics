"""Import the SFRoom2 textures and hook them up to the imported meshes.

Run after import_sfroom2.py, from the Unreal Output Log (Cmd -> Python):
    py "E:/GitHub/Graphics/Projects/Portfolio/SFRoom2/UE/Import/apply_textures_sfroom2.py"

materials.json (written from Blender) maps each Blender material to its texture files.
Builds one master material (M_SFRoom2_PBR) and a material instance per Blender material,
then assigns the instances to the mesh slots with the same name. Re-runnable.
"""
import json
import os
import re

import unreal

IMPORT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Game/SFRoom2"
TEXTURE_DIR = ROOT + "/Textures"
MATERIAL_DIR = ROOT + "/Materials"
MESH_DIR = ROOT + "/Meshes"
MASTER_PATH = MATERIAL_DIR + "/M_SFRoom2_PBR"

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
mel = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary

with open(os.path.join(IMPORT_DIR, "materials.json")) as f:
    materials = json.load(f)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def key(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


# --- Textures ---------------------------------------------------------------

def import_texture(path, channel):
    name = "T_" + safe_name(os.path.splitext(os.path.basename(path))[0])
    asset_path = TEXTURE_DIR + "/" + name
    if not eal.does_asset_exist(asset_path):
        task = unreal.AssetImportTask()
        task.filename = path
        task.destination_path = TEXTURE_DIR
        task.destination_name = name
        task.automated = True
        task.replace_existing = True
        task.save = False
        asset_tools.import_asset_tasks([task])
        if not eal.does_asset_exist(asset_path) and task.imported_object_paths:
            asset_path = task.imported_object_paths[0]
    tex = unreal.load_asset(asset_path)
    if channel == "Normal":
        # Substance "Principled BSDF" exports OpenGL normals; Unreal expects DirectX.
        tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
        tex.set_editor_property("srgb", False)
        tex.set_editor_property("flip_green_channel", True)
    elif channel in ("Roughness", "Metallic"):
        tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_DEFAULT)
        tex.set_editor_property("srgb", False)
    else:
        tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_DEFAULT)
        tex.set_editor_property("srgb", True)
    eal.save_loaded_asset(tex)
    return tex


textures = {}  # material name -> {channel: Texture2D}
with unreal.ScopedSlowTask(len(materials), "Importing SFRoom2 textures") as slow_task:
    slow_task.make_dialog(True)
    for mat_name, info in materials.items():
        slow_task.enter_progress_frame(1, mat_name)
        textures[mat_name] = {ch: import_texture(p, ch) for ch, p in info["textures"].items()}


def first_texture(channel):
    return next(t[channel] for t in textures.values() if channel in t)


# --- Master material ----------------------------------------------------------

def build_master():
    if eal.does_asset_exist(MASTER_PATH):
        mat = unreal.load_asset(MASTER_PATH)
        mel.delete_all_material_expressions(mat)
    else:
        mat = asset_tools.create_asset("M_SFRoom2_PBR", MATERIAL_DIR, unreal.Material, unreal.MaterialFactoryNew())

    def node(cls, x, y):
        return mel.create_material_expression(mat, cls, x, y)

    def scalar(name, value, x, y):
        n = node(unreal.MaterialExpressionScalarParameter, x, y)
        n.set_editor_property("parameter_name", name)
        n.set_editor_property("default_value", value)
        return n

    def sampler(name, channel, sampler_type, x, y):
        n = node(unreal.MaterialExpressionTextureSampleParameter2D, x, y)
        n.set_editor_property("parameter_name", name)
        n.set_editor_property("texture", first_texture(channel))
        n.set_editor_property("sampler_type", sampler_type)
        mel.connect_material_expressions(uv, "", n, "UVs")
        return n

    tiling = scalar("Tiling", 1.0, -1400, 0)
    texcoord = node(unreal.MaterialExpressionTextureCoordinate, -1400, -120)
    uv = node(unreal.MaterialExpressionMultiply, -1200, 0)
    mel.connect_material_expressions(texcoord, "", uv, "A")
    mel.connect_material_expressions(tiling, "", uv, "B")

    base = sampler("BaseColor", "BaseColor", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, -900, -400)
    mel.connect_material_property(base, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)

    metal = sampler("Metallic", "Metallic", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -900, -100)
    mel.connect_material_property(metal, "R", unreal.MaterialProperty.MP_METALLIC)

    # Roughness remap = saturate((r - RoughnessMin) / (RoughnessMax - RoughnessMin)), mirrors Blender's ColorRamp.
    rough = sampler("Roughness", "Roughness", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -900, 200)
    rmin = scalar("RoughnessMin", 0.0, -900, 450)
    rmax = scalar("RoughnessMax", 1.0, -900, 550)
    num = node(unreal.MaterialExpressionSubtract, -600, 250)
    mel.connect_material_expressions(rough, "R", num, "A")
    mel.connect_material_expressions(rmin, "", num, "B")
    den = node(unreal.MaterialExpressionSubtract, -600, 450)
    mel.connect_material_expressions(rmax, "", den, "A")
    mel.connect_material_expressions(rmin, "", den, "B")
    div = node(unreal.MaterialExpressionDivide, -450, 300)
    mel.connect_material_expressions(num, "", div, "A")
    mel.connect_material_expressions(den, "", div, "B")
    sat = node(unreal.MaterialExpressionSaturate, -300, 300)
    mel.connect_material_expressions(div, "", sat, "")
    mel.connect_material_property(sat, "", unreal.MaterialProperty.MP_ROUGHNESS)

    normal = sampler("Normal", "Normal", unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL, -900, 700)
    flat = node(unreal.MaterialExpressionConstant3Vector, -600, 650)
    flat.set_editor_property("constant", unreal.LinearColor(0, 0, 1, 1))
    nstr = scalar("NormalStrength", 1.0, -600, 900)
    nlerp = node(unreal.MaterialExpressionLinearInterpolate, -300, 700)
    mel.connect_material_expressions(flat, "", nlerp, "A")
    mel.connect_material_expressions(normal, "RGB", nlerp, "B")
    mel.connect_material_expressions(nstr, "", nlerp, "Alpha")
    mel.connect_material_property(nlerp, "", unreal.MaterialProperty.MP_NORMAL)

    emissive = sampler("Emissive", "Emissive", unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, -900, 1000)
    estr = scalar("EmissiveStrength", 0.0, -600, 1150)
    emul = node(unreal.MaterialExpressionMultiply, -300, 1050)
    mel.connect_material_expressions(emissive, "RGB", emul, "A")
    mel.connect_material_expressions(estr, "", emul, "B")
    mel.connect_material_property(emul, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

    mel.recompile_material(mat)
    eal.save_loaded_asset(mat)
    return mat


master = build_master()


# --- Material instances -------------------------------------------------------

instances = {}  # key(material name) -> MaterialInstanceConstant
for mat_name, info in materials.items():
    name = "MI_" + safe_name(mat_name)
    path = MATERIAL_DIR + "/" + name
    if eal.does_asset_exist(path):
        mi = unreal.load_asset(path)
    else:
        mi = asset_tools.create_asset(name, MATERIAL_DIR, unreal.MaterialInstanceConstant,
                                      unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(mi, master)
    for channel, tex in textures[mat_name].items():
        mel.set_material_instance_texture_parameter_value(mi, channel, tex)
    has_emissive = "Emissive" in textures[mat_name]
    ramp = info.get("rough_ramp") or [0.0, 1.0]
    mel.set_material_instance_scalar_parameter_value(mi, "Tiling", info.get("tiling", 1.0))
    mel.set_material_instance_scalar_parameter_value(mi, "RoughnessMin", ramp[0])
    mel.set_material_instance_scalar_parameter_value(mi, "RoughnessMax", ramp[1])
    mel.set_material_instance_scalar_parameter_value(mi, "NormalStrength", info.get("normal_strength", 1.0))
    mel.set_material_instance_scalar_parameter_value(
        mi, "EmissiveStrength", info.get("emissive_strength", 0.0) if has_emissive else 0.0)
    mel.update_material_instance(mi)
    eal.save_loaded_asset(mi)
    instances[key(mat_name)] = mi


# --- Assign to meshes ---------------------------------------------------------

assigned, unmatched = 0, set()
for asset_path in eal.list_assets(MESH_DIR, recursive=True, include_folder=False):
    mesh = unreal.load_asset(asset_path)
    if not isinstance(mesh, unreal.StaticMesh):
        continue
    changed = False
    for i, slot in enumerate(mesh.get_editor_property("static_materials")):
        slot_name = str(slot.get_editor_property("material_slot_name"))
        mi = instances.get(key(slot_name))
        if mi is None:
            unmatched.add(slot_name)
            continue
        mesh.set_material(i, mi)
        assigned += 1
        changed = True
    if changed:
        eal.save_loaded_asset(mesh)

unreal.log("SFRoom2 textures done: {} instances, {} slots assigned, unmatched slots: {}".format(
    len(instances), assigned, sorted(unmatched)))
