import os
import sys
import math

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

# ----------------- 可调参数 -----------------

# 最小可操作平面尺寸（单位：米）
MIN_WORKSPACE_WIDTH = 0.40   # 40 cm
MIN_WORKSPACE_DEPTH = 0.40   # 50 cm

# function_space / disturb_space 尺寸相对于平面 bbox 的比例
FUNCTION_SCALE = 0.7
DISTURB_SCALE = 0.8

# function_space / disturb_space 的最小尺寸（单位：米）
MIN_SUBSPACE_SIZE = 0.05  # 5 cm

# Robot base footprint 尺寸
ROBOT_LOC_SIZE_X = 0.40   # 40 cm
ROBOT_LOC_SIZE_Y = 0.40   # 40 cm

# Robot 相对 workspace 的偏移（沿 Y 方向），单位米
ROBOT_OFFSET_Y = 0.2      # 从工作区域边缘往外偏 0.2m

# 厚度阈值，用于过滤墙面等（Z 方向 thickness）
MAX_PLANE_THICKNESS = 2  # 1m

# 可操作平面高度范围（台面高度）
MIN_Z_TOP = 0.6   # 0.5 m
MAX_Z_TOP = 1.2   # 1.1 m

# 工作区块厚度（cube 的高度），单位米
FUNCTION_THICKNESS = 0.03   # 5 cm
DISTURB_THICKNESS  = 0.03   # 5 cm
ROBOT_THICKNESS    = 0.01   # 2 cm

# 放在平面上方的小抬升，避免 z fighting
PLANE_EPS_Z = 0.005  # 0.5 cm

# 要排除的名字关键字（不要地毯、地面等）
EXCLUDE_NAME_KEYWORDS = [
    "carpet", "rug", "floor", "ground","sofa","chair"
]

ENABLE_ROOM_Y_FILTER = True
# 房间 y 聚类时，给出的额外 margin（米）
ROOM_Y_MARGIN = 15


# ----------------- 基础工具 -----------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_excluded_by_name(prim):
    path_str = prim.GetPath().pathString.lower()
    return any(kw in path_str for kw in EXCLUDE_NAME_KEYWORDS)


def rect_size_from_bbox(box: Gf.Range3d):
    """从 bbox 中解析 (cx,cy,z_top)、(sx,sy)、(min,max,sz)"""
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    size = box.GetSize()

    center = (min_pt + max_pt) * 0.5
    cx, cy, _ = center[0], center[1], center[2]
    sx, sy, sz = size[0], size[1], size[2]
    z_top = max_pt[2]

    return (cx, cy, z_top), (sx, sy), (min_pt, max_pt, sz)


def create_colored_box_mesh(stage, path, center, size_xyz, color, z_offset=0.0):
    """创建带颜色的长方体 mesh，用来画 workspace 立方块"""
    sx, sy, sz = size_xyz
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5

    mesh = UsdGeom.Mesh.Define(stage, path)

    points = [
        Gf.Vec3f(-hx, -hy, -hz),
        Gf.Vec3f( hx, -hy, -hz),
        Gf.Vec3f( hx,  hy, -hz),
        Gf.Vec3f(-hx,  hy, -hz),
        Gf.Vec3f(-hx, -hy,  hz),
        Gf.Vec3f( hx, -hy,  hz),
        Gf.Vec3f( hx,  hy,  hz),
        Gf.Vec3f(-hx,  hy,  hz),
    ]

    face_vertex_indices = [
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        3, 2, 6, 3, 6, 7,
        0, 5, 1, 0, 4, 5,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ]
    face_vertex_counts = [3] * 12

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)

    color_attr = mesh.CreateDisplayColorAttr()
    color_attr.Set([Gf.Vec3f(*color)])

    xform_api = UsdGeom.XformCommonAPI(mesh)
    x, y, z = center
    xform_api.SetTranslate(Gf.Vec3d(float(x), float(y), float(z + z_offset)))

    return mesh


# ----------------- Actor 结构分析 (核心修改) -----------------

def list_meshes_under_actor(actor_prim):
    """列出 Actor_xxx 子树里所有 Mesh"""
    meshes = []
    for child in Usd.PrimRange(actor_prim):
        if child == actor_prim:
            continue
        if child.IsA(UsdGeom.Mesh):
            meshes.append(child)
    return meshes


def get_actor_surface_info(actor_prim, bbox_cache):
    """
    【修复版】
    强制使用 Matrix 计算来解决 ComputeWorldBound 失效的问题。
    逻辑：
    1. 获取 Mesh 的 Local Range (几何体原始大小)。
    2. 获取 Mesh 的 LocalToWorld Transform Matrix (强制包含所有父级变换)。
    3. 将 Matrix 应用到 Local Range，手动计算出 World BBox。
    """
    meshes = list_meshes_under_actor(actor_prim)
    if not meshes:
        return None

    # 收集所有 Mesh 的 *手动计算后的* 世界边界框
    world_bboxes = []
    
    for mesh_prim in meshes:
        # 1. 获取局部包围盒 (纯几何数据，不含位置)
        # ComputeLocalBound 比 ComputeWorldBound 可靠，因为它只看点
        local_bound = bbox_cache.ComputeLocalBound(mesh_prim)
        local_box = local_bound.GetBox()
        
        if local_box.IsEmpty():
            continue

        # 2. 获取从 Mesh 到 World 的完整变换矩阵
        # 这会自动把 mesh -> Geom -> Actor -> ... -> World 的所有变换乘起来
        xformable = UsdGeom.Xformable(mesh_prim)
        # 使用 Default 时间码
        local_to_world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        # 3. 变换包围盒
        # Gf.BBox3d 可以帮我们将一个 AABB 乘以矩阵，算出新的 AABB
        # (注意：旋转后 AABB 会变大以包围物体)
        world_bbox_calc = Gf.BBox3d(local_box, local_to_world_transform)
        aligned_world_range = world_bbox_calc.ComputeAlignedRange()
        
        if not aligned_world_range.IsEmpty():
            world_bboxes.append(aligned_world_range)

    if not world_bboxes:
        return None

    # 4. 合并所有计算出的世界边界框
    overall_box = Gf.Range3d()
    for box in world_bboxes:
        overall_box.UnionWith(box)
    
    if overall_box.IsEmpty():
        return None
        
    # 5. 从整体边界框解析信息
    (cx, cy, z_top), (sx, sy), (min_pt, max_pt, sz) = rect_size_from_bbox(overall_box)

    return (cx, cy, z_top), (sx, sy), (min_pt, max_pt, sz)


def is_candidate_actor_surface(actor_prim, bbox_cache):
    """筛选逻辑"""
    if is_excluded_by_name(actor_prim):
        return None

    info = get_actor_surface_info(actor_prim, bbox_cache)
    if info is None:
        return None
    (cx, cy, z_top), (sx, sy), (_, _, sz) = info

    # 厚度过滤
    if sz > MAX_PLANE_THICKNESS:
        return None

    cond1 = (sx >= MIN_WORKSPACE_WIDTH and sy >= MIN_WORKSPACE_DEPTH)
    cond2 = (sy >= MIN_WORKSPACE_WIDTH and sx >= MIN_WORKSPACE_DEPTH)
    if not (cond1 or cond2):
        return None

    if not (MIN_Z_TOP <= z_top <= MAX_Z_TOP):
        return None

    return info


# ----------------- 主逻辑 -----------------

def generate_workspaces(scene_usd_path):
    stage = Usd.Stage.Open(scene_usd_path)
    if stage is None:
        print(f"[ERROR] Cannot open USD stage: {scene_usd_path}")
        return

    print(f"[INFO] Opened stage: {scene_usd_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy
        ]
    )

    scene_dir = os.path.dirname(scene_usd_path)
    workspace_dir = os.path.join(scene_dir, "workspace_meta")
    ensure_dir(workspace_dir)
    print(f"[INFO] Workspace output dir: {workspace_dir}")

    # -------- Room Y Filter --------
    room_y_samples = []
    if ENABLE_ROOM_Y_FILTER:
        for prim in stage.Traverse():
            if not prim.GetName().startswith("Actor_"):
                continue
            info = is_candidate_actor_surface(prim, bbox_cache)
            if info is None:
                continue
            (cx, cy, z_top), _, _ = info
            room_y_samples.append(cy)

    if ENABLE_ROOM_Y_FILTER and room_y_samples:
        room_center_y = sum(room_y_samples) / len(room_y_samples)
        room_min_y = room_center_y - ROOM_Y_MARGIN
        room_max_y = room_center_y + ROOM_Y_MARGIN
        print(f"[INFO] Estimated room y-range: "
              f"[{room_min_y:.3f}, {room_max_y:.3f}] based on {len(room_y_samples)} candidates")
    else:
        room_min_y = room_max_y = None
        print("[INFO] Room y-range filter skipped or failed.")

    # -------- Generate Workspaces --------
    workspace_count = 0

    for prim in stage.Traverse():
        name = prim.GetName()
        if not name.startswith("Actor_"):
            continue

        info = is_candidate_actor_surface(prim, bbox_cache)
        if info is None:
            continue

        (cx, cy, z_top), (sx, sy), (min_pt, max_pt, sz) = info

        if ENABLE_ROOM_Y_FILTER and room_min_y is not None:
            if not (room_min_y <= cy <= room_max_y):
                continue

        print(
            f"[INFO] Valid Workspace: {prim.GetPath()} | "
            f"Center=({cx:.3f}, {cy:.3f}, {z_top:.3f})"
        )

        # 计算 subspaces
        fx = sx * FUNCTION_SCALE
        fy = sy * FUNCTION_SCALE
        dx = sx * DISTURB_SCALE
        dy = sy * DISTURB_SCALE

        if fx < MIN_SUBSPACE_SIZE or fy < MIN_SUBSPACE_SIZE:
            continue
        if dx < MIN_SUBSPACE_SIZE or dy < MIN_SUBSPACE_SIZE:
            continue

        table_min_y = min_pt[1]
        table_max_y = max_pt[1]

        # RobotLocation
        robot_y = table_min_y - ROBOT_OFFSET_Y
        robot_center = (cx, robot_y, ROBOT_THICKNESS * 0.5)
        robot_size = (ROBOT_LOC_SIZE_X, ROBOT_LOC_SIZE_Y, ROBOT_THICKNESS)

        # FunctionSpace
        front_margin = 0.10
        fy = min(fy, (table_max_y - table_min_y) - 2 * front_margin)
        if fy < MIN_SUBSPACE_SIZE:
            continue

        func_center_y = table_min_y + front_margin + fy * 0.5
        func_bottom_z = z_top + PLANE_EPS_Z
        func_center_z = func_bottom_z + FUNCTION_THICKNESS * 0.5
        func_center = (cx, func_center_y, func_center_z)

        # DisturbSpace
        back_margin = 0.10
        dy = min(dy, (table_max_y - table_min_y) - 2 * back_margin)
        if dy < MIN_SUBSPACE_SIZE:
            continue

        dist_center_y = table_max_y - back_margin - dy * 0.5
        dist_bottom_z = z_top + PLANE_EPS_Z
        dist_center_z = dist_bottom_z + DISTURB_THICKNESS * 0.5
        dist_center = (cx, dist_center_y, dist_center_z)

        workspace_count += 1
        workspace_name = f"workspace_{workspace_count:03d}.usd"
        workspace_path = os.path.join(workspace_dir, workspace_name)

        ws_stage = Usd.Stage.CreateNew(workspace_path)
        ws_stage.SetDefaultPrim(ws_stage.DefinePrim("/Workspace", "Xform"))

        create_colored_box_mesh(
            ws_stage, "/Workspace/FunctionSpace", func_center, (fx, fy, FUNCTION_THICKNESS), (1.0, 0.0, 0.0)
        )
        create_colored_box_mesh(
            ws_stage, "/Workspace/DisturbSpace", dist_center, (dx, dy, DISTURB_THICKNESS), (0.0, 1.0, 0.0)
        )
        create_colored_box_mesh(
            ws_stage, "/Workspace/RobotLocation", robot_center, robot_size, (0.0, 0.0, 1.0)
        )

        ws_stage.GetRootLayer().Save()
        print(f"      -> Saved: {workspace_path}")

    print(f"[DONE] Generated {workspace_count} workspace files.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_workspace_annotator.py /path/to/scene.usd")
        sys.exit(1)

    scene_path = sys.argv[1]
    generate_workspaces(scene_path)