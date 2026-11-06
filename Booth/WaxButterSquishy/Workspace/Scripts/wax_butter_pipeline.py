"""WaxButterSquishy: Blender -> VRChat (Unity) asset pipeline.

Run inside Blender with WaxButterSquishy.blend open (Scripting tab, or the Python console):
    exec(open(r"E:/GitHub/Graphics/Booth/WaxButterSquishy/Workspace/Scripts/wax_butter_pipeline.py").read())
    run()                       # every step
    run(["fracture", "uv"])     # only some steps

Steps (in order):
    uv        box-style UV unwrap of 4ozButterBar and Wax (Wax leaves a strip free for fracture faces)
    fracture  Voronoi-fracture Wax into PIECE_COUNT pieces under the Wax_Fractured empty
    material  Butter / Wax materials (one material per object, pieces share Wax)
    bake      bake procedural textures with Cycles and write the Principled_BSDF / lilToon / Poiyomi sets
    export    WaxButterSquishy.fbx for Unity (VRChat)

The source objects (4ozButterBar, Wax) are never changed geometrically; everything is re-runnable.
"""
import bisect
import os
import random

import bmesh
import bpy
import numpy as np
from mathutils import Vector, noise
from mathutils.bvhtree import BVHTree

BUTTER_NAME = "4ozButterBar"
WAX_NAME = "Wax"
FRACTURE_ROOT = "Wax_Fractured"
PIECE_PREFIX = "Wax_Piece_"

# --- Fracture ----------------------------------------------------------------
PIECE_COUNT = 40
FRACTURE_SEED = 7            # change for a different crack pattern
SEED_CANDIDATES = 4          # Mitchell best-candidate: higher = more even piece sizes
MIN_PIECE_RATIO = 0.3        # reject a pattern if a piece is smaller than this x the mean volume
CRACK_NOISE_AMP = 0.006      # m, how far crack lines wander from straight Voronoi edges
CRACK_NOISE_FREQ = 8.0

# --- UV / textures -----------------------------------------------------------
TEX_SIZE = 2048
WAX_UV_REGION = (0.0, 0.13, 1.0, 1.0)        # u0, v0, u1, v1 for the wax surface islands
CUT_UV_REGION = (0.005, 0.005, 0.995, 0.115)  # strip shared by every fracture face
UV_MARGIN = 0.006
BAKE_MARGIN = 8

# sRGB hex; re-run "bake" after changing
WAX_COLOR = "#F2EEE6"
WAX_CUT_COLOR = "#E4DCCD"
BUTTER_COLOR = "#F6DF92"

WORKSPACE = bpy.path.abspath("//")
TEX_DIR = os.path.join(WORKSPACE, "Textures")
FBX_PATH = os.path.join(WORKSPACE, "WaxButterSquishy.fbx")

STEPS = ["uv", "fracture", "material", "bake", "export"]


# --- Helpers -----------------------------------------------------------------
def hex_to_linear(value):
    value = value.lstrip("#")
    srgb = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb) + (1.0,)


def object_mode():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def select_only(objects, active=None):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objects:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objects[0]


def pieces():
    root = bpy.data.objects.get(FRACTURE_ROOT)
    return sorted(root.children, key=lambda o: o.name) if root else []


def islands(bm):
    """Face islands of a bmesh, connected through shared edges."""
    bm.faces.ensure_lookup_table()
    seen = set()
    for face in bm.faces:
        if face.index in seen:
            continue
        stack, island = [face], []
        seen.add(face.index)
        while stack:
            f = stack.pop()
            island.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in seen:
                        seen.add(lf.index)
                        stack.append(lf)
        yield island


def volume_centroid(bm):
    center, six_vol = Vector(), 0.0
    for tri in bm.calc_loop_triangles():
        a, b, c = (l.vert.co for l in tri)
        d = a.dot(b.cross(c))
        six_vol += d
        center += (a + b + c) * d
    return center / (4 * six_vol) if six_vol else Vector()


# --- UV ----------------------------------------------------------------------
def unwrap_box(obj, region):
    """Seams where the dominant normal axis changes (6 patches per shell), angle-based unwrap,
    then pack the islands into region."""
    object_mode()
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)

    def side(f):
        axis = max(range(3), key=lambda i: abs(f.normal[i]))
        return axis * 2 + (f.normal[axis] > 0)

    for e in bm.edges:
        lf = e.link_faces
        e.seam = len(lf) == 2 and side(lf[0]) != side(lf[1])
    bm.to_mesh(me)
    bm.free()

    select_only([obj])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.0)

    # Squash everything into the target box, then pack inside its AABB.
    bm = bmesh.from_edit_mesh(me)
    uv = bm.loops.layers.uv.active
    loops = [l for f in bm.faces for l in f.loops]
    us = [l[uv].uv.x for l in loops]
    vs = [l[uv].uv.y for l in loops]
    u0, v0, u1, v1 = region
    for l in loops:
        l[uv].uv.x = u0 + (l[uv].uv.x - min(us)) / (max(us) - min(us)) * (u1 - u0)
        l[uv].uv.y = v0 + (l[uv].uv.y - min(vs)) / (max(vs) - min(vs)) * (v1 - v0)
    bmesh.update_edit_mesh(me)

    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(udim_source="ORIGINAL_AABB", rotate=True, rotate_method="CARDINAL",
                            scale=True, margin_method="FRACTION", margin=UV_MARGIN, shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")


def uv_density(obj):
    """UV units per metre of obj's current layout."""
    me = obj.data
    uv = me.uv_layers.active.data
    area_3d = sum(p.area for p in me.polygons)
    area_uv = 0.0
    for p in me.polygons:
        pts = [uv[i].uv for i in p.loop_indices]
        area_uv += abs(sum(pts[i].x * pts[i - 1].y - pts[i - 1].x * pts[i].y for i in range(len(pts)))) / 2
    return (area_uv / area_3d) ** 0.5


def step_uv():
    unwrap_box(bpy.data.objects[BUTTER_NAME], (0.0, 0.0, 1.0, 1.0))
    unwrap_box(bpy.data.objects[WAX_NAME], WAX_UV_REGION)
    return {"butter_islands_uv_density": round(uv_density(bpy.data.objects[BUTTER_NAME]), 3),
            "wax_uv_density": round(uv_density(bpy.data.objects[WAX_NAME]), 3)}


# --- Fracture ----------------------------------------------------------------
def crack_offset(p):
    return noise.noise_vector(p * CRACK_NOISE_FREQ + Vector((3.1, 7.7, 1.3)),
                              noise_basis="PERLIN_ORIGINAL") * CRACK_NOISE_AMP


def clear_fracture():
    root = bpy.data.objects.get(FRACTURE_ROOT)
    old = [o for o in bpy.data.objects if o.name.startswith(PIECE_PREFIX)]
    for o in old:
        me = o.data
        bpy.data.objects.remove(o)
        if me.users == 0:
            bpy.data.meshes.remove(me)
    if root:
        bpy.data.objects.remove(root)


def pick_seeds(src, rng):
    """PIECE_COUNT points on the outer wax surface, spread with Mitchell's best-candidate."""
    bm = bmesh.new()
    bm.from_mesh(src.data)
    tris = [t for t in bm.calc_loop_triangles()
            if t[0].face.normal.dot(t[0].face.calc_center_median()) > 0]
    tris = [tuple(l.vert.co.copy() for l in t) for t in tris]
    bm.free()
    cum, total = [], 0.0
    for a, b, c in tris:
        total += (b - a).cross(c - a).length / 2
        cum.append(total)

    def sample():
        a, b, c = tris[min(bisect.bisect(cum, rng.random() * total), len(tris) - 1)]
        s, t = rng.random(), rng.random()
        if s + t > 1:
            s, t = 1 - s, 1 - t
        return a + (b - a) * s + (c - a) * t

    seeds = [sample()]
    while len(seeds) < PIECE_COUNT:
        candidates = [sample() for _ in range(SEED_CANDIDATES)]
        seeds.append(max(candidates, key=lambda p: min((p - q).length for q in seeds)))
    return seeds


def voronoi_cell(seeds, i, bounds):
    """Convex Voronoi cell of seeds[i], clipped to bounds (half extents)."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * bounds.x, v.co.y * bounds.y, v.co.z * bounds.z))
    pi = seeds[i]
    for j, pj in enumerate(seeds):
        if j == i:
            continue
        n = (pj - pi).normalized()
        cut = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                                     plane_co=(pi + pj) / 2, plane_no=n, clear_outer=True)["geom_cut"]
        cut = [e for e in cut if isinstance(e, bmesh.types.BMEdge)]
        if cut:
            bmesh.ops.holes_fill(bm, edges=cut, sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("__cell")
    bm.to_mesh(me)
    bm.free()
    return me


def fracture_once(src, seed):
    """Returns a list of (bmesh, centroid, volume) per piece, or raises if the pattern is unusable."""
    rng = random.Random(seed)
    seeds = [p + crack_offset(p) for p in pick_seeds(src, rng)]

    # Cut a noise-warped copy with flat Voronoi cells, then unwarp: crack lines come out wavy.
    warp_me = src.data.copy()
    warp_me.name = "__wax_warp"
    for v in warp_me.vertices:
        v.co = v.co + crack_offset(v.co)
    warp_me.attributes.new("wax_surface", "INT", "FACE").data.foreach_set("value", [1] * len(warp_me.polygons))
    tmp = bpy.data.collections.new("__fracture_tmp")
    bpy.context.scene.collection.children.link(tmp)
    warp_obj = bpy.data.objects.new("__wax_warp", warp_me)
    cell_obj = bpy.data.objects.new("__cell", bpy.data.meshes.new("__cell"))
    tmp.objects.link(warp_obj)
    tmp.objects.link(cell_obj)
    mod = warp_obj.modifiers.new("Fracture", "BOOLEAN")
    mod.operation, mod.solver, mod.object = "INTERSECT", "EXACT", cell_obj

    bounds = Vector(src.dimensions) / 2 + Vector((0.08, 0.08, 0.08))
    raw = []
    try:
        for i in range(len(seeds)):
            old = cell_obj.data
            cell_obj.data = voronoi_cell(seeds, i, bounds)
            bpy.data.meshes.remove(old)
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            raw.append(bpy.data.meshes.new_from_object(warp_obj.evaluated_get(dg)))
    finally:
        bpy.data.objects.remove(warp_obj)
        bpy.data.objects.remove(cell_obj)
        bpy.data.collections.remove(tmp)
        for me in (warp_me, *[m for m in bpy.data.meshes if m.name.startswith("__cell") and m.users == 0]):
            bpy.data.meshes.remove(me)

    src_bm = bmesh.new()
    src_bm.from_mesh(src.data)
    bvh = BVHTree.FromBMesh(src_bm)
    src_bm.free()

    result = []
    try:
        for me in raw:
            bm = bmesh.new()
            bm.from_mesh(me)
            for v in bm.verts:
                # Undo the warp (fixed-point inverse), then snap onto the real wax surface.
                p = v.co.copy()
                for _ in range(20):
                    p = v.co - crack_offset(p)
                v.co = bvh.find_nearest(p)[0]
            parts = list(islands(bm))
            if len(parts) != 1:
                bm.free()
                raise ValueError(f"cell produced {len(parts)} islands")
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
            if not all(e.is_manifold for e in bm.edges):
                bm.free()
                raise ValueError("non-manifold piece")
            result.append((bm, volume_centroid(bm), bm.calc_volume(signed=True)))
    except ValueError:
        for bm, _, _ in result:
            bm.free()
        raise
    finally:
        for me in raw:
            bpy.data.meshes.remove(me)

    mean = sum(r[2] for r in result) / len(result)
    if min(r[2] for r in result) < mean * MIN_PIECE_RATIO:
        for bm, _, _ in result:
            bm.free()
        raise ValueError("a piece is too small")
    return result


def uv_cut_faces(bm, density, rng):
    """Fracture faces (no wax_surface flag) get a planar projection at the surface texel density,
    dropped at a random spot inside the shared cut strip."""
    uv = bm.loops.layers.uv.active
    flag = bm.faces.layers.int.get("wax_surface")
    u0, v0, u1, v1 = CUT_UV_REGION
    cut = [f for f in bm.faces if not f[flag]]
    for f in cut:
        n = f.normal
        t = max((e.verts[1].co - e.verts[0].co for e in f.edges), key=lambda d: d.length).normalized()
        b = n.cross(t)
        o = f.verts[0].co
        pts = [((l.vert.co - o).dot(t), (l.vert.co - o).dot(b)) for l in f.loops]
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        w, h = max(p[0] for p in pts) - x0, max(p[1] for p in pts) - y0
        k = min(density, (u1 - u0) / max(w, 1e-6), (v1 - v0) / max(h, 1e-6))
        ou = rng.uniform(u0, u1 - w * k)
        ov = rng.uniform(v0, v1 - h * k)
        for l, (x, y) in zip(f.loops, pts):
            l[uv].uv = (ou + (x - x0) * k, ov + (y - y0) * k)
    return len(cut)


def step_fracture():
    object_mode()
    src = bpy.data.objects[WAX_NAME]
    clear_fracture()
    density = uv_density(src)

    for attempt in range(20):
        try:
            parts = fracture_once(src, FRACTURE_SEED + attempt)
            break
        except ValueError as err:
            print(f"fracture seed {FRACTURE_SEED + attempt} rejected: {err}")
    else:
        raise RuntimeError("no usable fracture pattern in 20 tries")

    root = bpy.data.objects.new(FRACTURE_ROOT, None)
    root.empty_display_type, root.empty_display_size = "PLAIN_AXES", 0.1
    collection = src.users_collection[0]
    collection.objects.link(root)

    rng = random.Random(FRACTURE_SEED)
    parts.sort(key=lambda p: (round(p[1].x, 2), p[1].z, p[1].y))
    cut_faces = 0
    for k, (bm, center, _) in enumerate(parts, 1):
        cut_faces += uv_cut_faces(bm, density, rng)
        bmesh.ops.translate(bm, verts=bm.verts, vec=-center)
        name = f"{PIECE_PREFIX}{k:02d}"
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        me.attributes.remove(me.attributes["wax_surface"])
        for mat in src.data.materials:
            me.materials.append(mat)
        obj = bpy.data.objects.new(name, me)
        obj.location = center
        collection.objects.link(obj)
        obj.parent = root
        # The intact Wax is the default look; pieces sit exactly on top of it, so keep them hidden.
        obj.hide_set(True)
        obj.hide_render = True

    src.hide_set(False)
    src.hide_render = False
    vols = [p[2] for p in parts]
    return {"seed": FRACTURE_SEED + attempt, "pieces": len(parts), "cut_faces": cut_faces,
            "volume_min_max_ratio": round(min(vols) / max(vols), 3)}


# --- Materials ---------------------------------------------------------------
def node(nt, kind, x, y, **inputs):
    n = nt.nodes.new(kind)
    n.location = (x, y)
    for key, val in inputs.items():
        n.inputs[key].default_value = val
    return n


def madd(nt, socket, mul, add, x, y):
    n = node(nt, "ShaderNodeMath", x, y)
    n.operation = "MULTIPLY_ADD"
    nt.links.new(socket, n.inputs[0])
    n.inputs[1].default_value, n.inputs[2].default_value = mul, add
    return n.outputs[0]


def add(nt, a, b, x, y):
    n = node(nt, "ShaderNodeMath", x, y)
    n.operation = "ADD"
    nt.links.new(a, n.inputs[0])
    nt.links.new(b, n.inputs[1])
    return n.outputs[0]


def procedural(nt, kind):
    """Builds the look of one surface. Returns (color, roughness, height) sockets, height ~0..1."""
    links = nt.links
    coord = node(nt, "ShaderNodeTexCoord", -1400, 0).outputs["Object"]
    big = node(nt, "ShaderNodeTexNoise", -1150, 300, Scale=3.0, Detail=2.0)
    fine = node(nt, "ShaderNodeTexNoise", -1150, 0, Scale=90.0, Detail=8.0, Roughness=0.6)
    for n in (big, fine):
        links.new(coord, n.inputs["Vector"])

    base = {"wax": WAX_COLOR, "wax_cut": WAX_CUT_COLOR, "butter": BUTTER_COLOR}[kind]
    lin = hex_to_linear(base)
    ramp = node(nt, "ShaderNodeValToRGB", -850, 400)
    ramp.color_ramp.elements[0].color = tuple(c * 0.9 for c in lin[:3]) + (1.0,)
    ramp.color_ramp.elements[1].color = tuple(min(1.0, c * 1.04) for c in lin[:3]) + (1.0,)
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.3, 0.7
    links.new(big.outputs["Fac"], ramp.inputs["Fac"])

    if kind == "wax":
        # Dipped wax: soft horizontal drip layers plus a little grain.
        wave = node(nt, "ShaderNodeTexWave", -1150, -300, Scale=2.5, Distortion=6.0, Detail=3.0,
                    **{"Detail Scale": 1.5})
        wave.wave_type, wave.bands_direction = "BANDS", "Z"
        links.new(coord, wave.inputs["Vector"])
        height = add(nt, madd(nt, wave.outputs["Fac"], 0.35, 0.3, -850, -300),
                     madd(nt, fine.outputs["Fac"], 0.3, -0.15, -850, -100), -650, -200)
        rough = madd(nt, fine.outputs["Fac"], 0.25, 0.2, -650, 100)
    elif kind == "wax_cut":
        height = madd(nt, fine.outputs["Fac"], 1.0, 0.0, -850, -100)
        rough = madd(nt, fine.outputs["Fac"], 0.2, 0.45, -650, 100)
    else:
        height = add(nt, madd(nt, fine.outputs["Fac"], 0.6, 0.0, -850, -100),
                     madd(nt, big.outputs["Fac"], 0.4, 0.0, -850, -300), -650, -200)
        rough = madd(nt, fine.outputs["Fac"], 0.3, 0.35, -650, 100)
    return ramp.outputs["Color"], rough, height


def bake_material(name, kind):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    color, rough, height = procedural(nt, kind)
    bump = node(nt, "ShaderNodeBump", -400, -200, Strength=0.6, Distance=0.002)
    nt.links.new(height, bump.inputs["Height"])
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", -150, 200)
    nt.links.new(color, bsdf.inputs["Base Color"])
    nt.links.new(rough, bsdf.inputs["Roughness"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    node(nt, "ShaderNodeEmission", -150, -250)
    node(nt, "ShaderNodeOutputMaterial", 150, 0)
    node(nt, "ShaderNodeTexImage", 150, -300).name = "BakeTarget"
    return mat


def wire_bake(mat, bake_pass, image):
    """Point a bake material at one pass: emission carries color/roughness/height."""
    nt = mat.node_tree
    out = nt.nodes["Material Output"]
    emit = nt.nodes["Emission"]
    bsdf = nt.nodes["Principled BSDF"]
    for l in list(out.inputs["Surface"].links) + list(emit.inputs["Color"].links):
        nt.links.remove(l)
    if bake_pass in ("normal", "ao"):
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    else:
        src = {"color": bsdf.inputs["Base Color"], "rough": bsdf.inputs["Roughness"],
               "height": nt.nodes["Bump"].inputs["Height"]}[bake_pass].links[0].from_socket
        nt.links.new(src, emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    target = nt.nodes["BakeTarget"]
    target.image = image
    nt.nodes.active = target


def final_material(name):
    """Principled material driven by the baked Principled_BSDF textures (what the FBX carries)."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", 0, 0, Metallic=0.0)
    out = node(nt, "ShaderNodeOutputMaterial", 300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def tex(suffix, y, non_color):
        path = os.path.join(TEX_DIR, "Principled_BSDF", f"{name}_{suffix}.png")
        img = bpy.data.images.load(path, check_existing=True)
        img.reload()
        img.filepath = bpy.path.relpath(path)
        img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
        n = node(nt, "ShaderNodeTexImage", -600, y)
        n.image = img
        return n

    if os.path.exists(os.path.join(TEX_DIR, "Principled_BSDF", f"{name}_BaseColor.png")):
        nt.links.new(tex("BaseColor", 300, False).outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(tex("Roughness", 0, True).outputs["Color"], bsdf.inputs["Roughness"])
        nmap = node(nt, "ShaderNodeNormalMap", -250, -300)
        nt.links.new(tex("Normal", -300, True).outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        bsdf.inputs["Base Color"].default_value = hex_to_linear(WAX_COLOR if name == "Wax" else BUTTER_COLOR)
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def step_material():
    butter = final_material("Butter")
    wax = final_material("Wax")
    assign(bpy.data.objects[BUTTER_NAME], butter)
    for obj in [bpy.data.objects[WAX_NAME], *pieces()]:
        assign(obj, wax)
    return {"materials": ["Butter", "Wax"], "wax_users": 1 + len(pieces())}


# --- Bake --------------------------------------------------------------------
PASSES = {  # pass: (bake type, colorspace, samples)
    "color": ("EMIT", "sRGB", 4),
    "rough": ("EMIT", "Non-Color", 4),
    "height": ("EMIT", "Non-Color", 4),
    "normal": ("NORMAL", "Non-Color", 8),
    "ao": ("AO", "Non-Color", 128),
}


def cut_strip_plane():
    """Plane whose UVs cover CUT_UV_REGION, so the shared fracture strip gets a wax-interior bake."""
    u0, v0, u1, v1 = CUT_UV_REGION
    pad = 0.004
    u0, v0, u1, v1 = u0 - pad, v0 - pad, u1 + pad, v1 + pad
    w = (u1 - u0) / uv_density(bpy.data.objects[WAX_NAME])
    h = w * (v1 - v0) / (u1 - u0)
    me = bpy.data.meshes.new("__cut_strip")
    me.from_pydata([(0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0)], [], [(0, 1, 2, 3)])
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", [u0, v0, u1, v0, u1, v1, u0, v1])
    obj = bpy.data.objects.new("__cut_strip", me)
    obj.location = (5.0, 0.0, 0.0)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def pixels(img):
    w, h = img.size
    arr = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    return arr.reshape(h, w, 4)


def save_png(arr, path, colorspace):
    h, w = arr.shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = bpy.data.images.new("__out", w, h, alpha=True)
    img.colorspace_settings.name = colorspace
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    out = np.ones((h, w, 4), dtype=np.float32)
    out[..., :arr.shape[2]] = arr
    img.pixels.foreach_set(np.clip(out, 0, 1).ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def write_texture_sets(name, baked):
    """Same file layout as HalloweenHat: Principled_BSDF / lilToon / Poiyomi."""
    color = baked["color"][..., :3]
    rough = baked["rough"][..., 0]
    smooth = 1.0 - rough
    metal = np.zeros_like(rough)
    normal = baked["normal"][..., :3]
    height = baked["height"][..., 0]
    ao = baked["ao"][..., 0]

    pb = os.path.join(TEX_DIR, "Principled_BSDF")
    save_png(color, os.path.join(pb, f"{name}_BaseColor.png"), "sRGB")
    save_png(rough, os.path.join(pb, f"{name}_Roughness.png"), "Non-Color")
    save_png(metal, os.path.join(pb, f"{name}_Metallic.png"), "Non-Color")
    save_png(normal, os.path.join(pb, f"{name}_Normal.png"), "Non-Color")
    save_png(height, os.path.join(pb, f"{name}_Displacement.png"), "Non-Color")

    lil = os.path.join(TEX_DIR, "lilToon")
    save_png(color, os.path.join(lil, f"{name}_lilToon_MainTex.png"), "sRGB")
    save_png(ao, os.path.join(lil, f"{name}_lilToon_AO.png"), "Non-Color")
    save_png(metal, os.path.join(lil, f"{name}_lilToon_Metallic.png"), "Non-Color")
    save_png(normal, os.path.join(lil, f"{name}_lilToon_Normal.png"), "Non-Color")
    save_png(smooth, os.path.join(lil, f"{name}_lilToon_Smoothness.png"), "Non-Color")

    poi = os.path.join(TEX_DIR, "Poiyomi")
    ms = np.stack([metal, smooth, np.ones_like(rough), np.ones_like(rough)], axis=-1)
    save_png(color, os.path.join(poi, f"t_{name}_BaseColor.png"), "sRGB")
    save_png(ao, os.path.join(poi, f"t_{name}_AmbientOcclusion.png"), "Non-Color")
    save_png(ms, os.path.join(poi, f"t_{name}_MetallicSmoothnessMaps.png"), "Non-Color")
    save_png(normal, os.path.join(poi, f"t_{name}_NormalMap.png"), "Non-Color")


def bake_set(name, jobs):
    """jobs: list of (object, bake material). All objects bake into the same images."""
    scene = bpy.context.scene
    baked = {}
    for bake_pass, (bake_type, colorspace, samples) in PASSES.items():
        img = bpy.data.images.new(f"__bake_{name}_{bake_pass}", TEX_SIZE, TEX_SIZE, alpha=False)
        img.colorspace_settings.name = colorspace
        img.generated_color = (0.5, 0.5, 1.0, 1.0) if bake_pass == "normal" else (1.0, 1.0, 1.0, 1.0)
        for obj, mat in jobs:
            assign(obj, mat)
            wire_bake(mat, bake_pass, img)
        scene.cycles.samples = samples
        select_only([o for o, _ in jobs])
        bpy.ops.object.bake(type=bake_type, margin=BAKE_MARGIN, use_clear=False,
                            normal_space="TANGENT", normal_r="POS_X", normal_g="POS_Y", normal_b="POS_Z")
        baked[bake_pass] = pixels(img)
        bpy.data.images.remove(img)
    write_texture_sets(name, baked)


def step_bake():
    object_mode()
    scene = bpy.context.scene
    butter, wax = bpy.data.objects[BUTTER_NAME], bpy.data.objects[WAX_NAME]
    others = [o for o in scene.objects if o not in (butter, wax)]
    saved = {
        "engine": scene.render.engine, "device": scene.cycles.device, "samples": scene.cycles.samples,
        "ao_distance": scene.world.light_settings.distance if scene.world else None,
        "hide_render": {o.name: o.hide_render for o in scene.objects},
        "materials": {o.name: list(o.data.materials) for o in (butter, wax)},
    }
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.light_settings.distance = 0.03

    strip = cut_strip_plane()
    mats = {k: bake_material(f"__Bake_{k}", k) for k in ("wax", "wax_cut", "butter")}
    try:
        for o in others:
            o.hide_render = True
        # Wax: shell + the shared fracture strip. Butter hidden so it does not darken the inner shell AO.
        butter.hide_render = True
        strip.hide_render = False
        bake_set("Wax", [(wax, mats["wax"]), (strip, mats["wax_cut"])])
        wax.hide_render = True
        strip.hide_render = True
        butter.hide_render = False
        bake_set("Butter", [(butter, mats["butter"])])
    finally:
        bpy.data.objects.remove(strip)
        bpy.data.meshes.remove(bpy.data.meshes["__cut_strip"])
        for m in mats.values():
            bpy.data.materials.remove(m)
        for o in scene.objects:
            o.hide_render = saved["hide_render"].get(o.name, False)
        for obj in (butter, wax):
            obj.data.materials.clear()
            for m in saved["materials"][obj.name]:
                obj.data.materials.append(m)
        scene.render.engine = saved["engine"]
        scene.cycles.device = saved["device"]
        scene.cycles.samples = saved["samples"]
        if saved["ao_distance"] is not None:
            scene.world.light_settings.distance = saved["ao_distance"]
    # Re-point the final materials at the fresh files.
    step_material()
    return {"textures": sorted(os.listdir(os.path.join(TEX_DIR, "Principled_BSDF")))}


# --- Export ------------------------------------------------------------------
def step_export():
    """FBX for Unity: Y-up, transforms baked so every object imports with rotation 0 / scale 1."""
    object_mode()
    objs = [bpy.data.objects[BUTTER_NAME], bpy.data.objects[WAX_NAME], bpy.data.objects[FRACTURE_ROOT], *pieces()]
    hidden = {o.name: o.hide_get() for o in objs}
    select_only(objs, active=objs[0])
    try:
        bpy.ops.export_scene.fbx(
            filepath=FBX_PATH, use_selection=True, object_types={"EMPTY", "MESH"},
            apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL", global_scale=1.0,
            axis_forward="-Z", axis_up="Y", bake_space_transform=True,
            use_mesh_modifiers=True, mesh_smooth_type="FACE", use_triangles=True, use_tspace=False,
            use_custom_props=False, add_leaf_bones=False, bake_anim=False,
            path_mode="AUTO", embed_textures=False)
    finally:
        for o in objs:
            o.hide_set(hidden[o.name])
    return {"fbx": FBX_PATH, "objects": len(objs)}


def run(steps=None):
    results = {}
    for step in steps or STEPS:
        results[step] = globals()[f"step_{step}"]()
        print(step, results[step])
    return results
