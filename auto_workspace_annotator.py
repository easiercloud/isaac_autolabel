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

# Robot 相对 workspace 的偏移距离（单位米）
# 以前是 ROBOT_OFFSET_Y，现在通用为偏置距离
ROBOT_OFFSET_DISTANCE = 0.15  # 15 cm

# 厚度阈值，用于过滤墙面等（Z 方向 thickness）
MAX_PLANE_THICKNESS = 2  # 1m

# 可操作平面高度范围（台面高度）
MIN_Z_TOP = 0.6   # 0.5 m
MAX_Z_TOP = 1.2   # 1.1 m

# 工作区块厚度（cube 的高度），单位米
FUNCTION_THICKNESS = 0.03   # 3 cm
DISTURB_THICKNESS  = 0.03   # 3 cm
ROBOT_THICKNESS    = 0.01   # 1 cm

# 放在平面上方的小抬升，避免 z fighting
PLANE_EPS_Z = 0.005  # 0.5 cm

# 要排除的名字关键字（不要地毯、地面等）
EXCLUDE_NAME_KEYWORDS = [
    "carpet", "rug", "floor", "ground", "wall", "ceiling","sofa","chair"
]

# 认为是“大物体”的最小投影面积（XY），小于这个当作小摆件，忽略其对 workspace 的阻挡
MIN_LARGE_OBJECT_AREA = 0.2* 0.2   # 0.09 m^2

# 大物体与 workspace 的交叠体积超过多少比例才认为是“致命碰撞”
# (Intersection Volume / Workspace Cube Volume)
COLLISION_RATIO_THRESHOLD = 0.3  # 30% 容忍度

ENABLE_ROOM_Y_FILTER = True
# 房间 y 聚类时，给出的额外 margin（米）
ROOM_Y_MARGIN = 15


# ----------------- 基础工具 -----------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_excluded_by_name(path_string):
    """通过名字判断是否是需要排除的地板/地毯等"""
    s = path_string.lower()
    return any(kw in s for kw in EXCLUDE_NAME_KEYWORDS)


def get_prim_world_bbox(prim, bbox_cache):
    """
    计算任意 Prim (Mesh/Xform) 的精确世界包围盒
    使用矩阵乘法确保父级变换被正确应用
    """
    # 1. 获取局部包围盒
    local_bound = bbox_cache.ComputeLocalBound(prim)
    local_box = local_bound.GetBox()
    
    if local_box.IsEmpty():
        return None

    # 2. 获取从 Prim 到 World 的完整变换矩阵
    xformable = UsdGeom.Xformable(prim)
    local_to_world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # 3. 变换包围盒
    world_bbox_calc = Gf.BBox3d(local_box, local_to_world_transform)
    aligned_world_range = world_bbox_calc.ComputeAlignedRange()
    
    return aligned_world_range


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
    """创建带颜色的长方体 mesh"""
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


# ----------------- 碰撞检测工具 -----------------

def get_bbox_intersection_volume(box_a: Gf.Range3d, box_b: Gf.Range3d):
    """计算两个 AABB 的重叠体积"""
    intersection = Gf.Range3d(box_a)
    intersection.IntersectWith(box_b)
    
    if intersection.IsEmpty():
        return 0.0
    
    size = intersection.GetSize()
    return size[0] * size[1] * size[2]


def check_collision_with_obstacles(target_center, target_size, obstacles, self_actor_path, threshold=COLLISION_RATIO_THRESHOLD):
    """
    检查目标区域（center, size）是否与 obstacle_list 中的物体发生碰撞
    :param obstacles: List of (prim_path_str, Gf.Range3d)
    :param self_actor_path: 当前正在处理的桌子的路径，需要排除自身碰撞
    :return: True if collision detected, False otherwise
    """
    cx, cy, cz = target_center
    sx, sy, sz = target_size
    
    # 构建目标的 BBox
    min_pt = Gf.Vec3d(cx - sx/2, cy - sy/2, cz - sz/2)
    max_pt = Gf.Vec3d(cx + sx/2, cy + sy/2, cz + sz/2)
    target_box = Gf.Range3d(min_pt, max_pt)
    target_volume = sx * sy * sz
    
    if target_volume <= 0:
        return False

    for obs_path, obs_box in obstacles:
        # 1. 排除自己 (如果是当前 Actor 的一部分)
        if obs_path.startswith(self_actor_path):
            continue
            
        # 2. 计算重叠
        inter_vol = get_bbox_intersection_volume(target_box, obs_box)
        
        if inter_vol > 0:
            ratio = inter_vol / target_volume
            if ratio > threshold:
                # print(f"   [COLLISION] Hit {obs_path}, Ratio: {ratio:.2f}")
                return True
                
    return False


# ----------------- Actor 分析逻辑 -----------------

def list_meshes_under_actor(actor_prim):
    meshes = []
    for child in Usd.PrimRange(actor_prim):
        if child == actor_prim:
            continue
        if child.IsA(UsdGeom.Mesh):
            meshes.append(child)
    return meshes


def get_actor_surface_info(actor_prim, bbox_cache):
    """
    获取 Actor 的整体世界 BBox
    """
    meshes = list_meshes_under_actor(actor_prim)
    if not meshes:
        return None

    world_bboxes = []
    for mesh_prim in meshes:
        # 复用 get_prim_world_bbox 逻辑
        wb = get_prim_world_bbox(mesh_prim, bbox_cache)
        if wb and not wb.IsEmpty():
            world_bboxes.append(wb)

    if not world_bboxes:
        return None

    overall_box = Gf.Range3d()
    for box in world_bboxes:
        overall_box.UnionWith(box)
    
    if overall_box.IsEmpty():
        return None
        
    (cx, cy, z_top), (sx, sy), (min_pt, max_pt, sz) = rect_size_from_bbox(overall_box)
    return (cx, cy, z_top), (sx, sy), (min_pt, max_pt, sz)


def is_candidate_actor_surface(actor_prim, bbox_cache):
    if is_excluded_by_name(actor_prim.GetPath().pathString):
        return None

    info = get_actor_surface_info(actor_prim, bbox_cache)
    if info is None:
        return None
    (cx, cy, z_top), (sx, sy), (_, _, sz) = info

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
    if not stage:
        print(f"[ERROR] Cannot open USD stage: {scene_usd_path}")
        return

    print(f"[INFO] Opened stage: {scene_usd_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    )

    scene_dir = os.path.dirname(scene_usd_path)
    workspace_dir = os.path.join(scene_dir, "workspace_meta")
    ensure_dir(workspace_dir)

    # ==========================================
    # 1. 预计算所有潜在障碍物 (Obstacles)
    # ==========================================
    print("[INFO] Building obstacle list...")
    all_obstacles = [] # list of (path_str, Gf.Range3d)
    
    for prim in stage.Traverse():
        # 我们只关心 Mesh 作为障碍物
        if not prim.IsA(UsdGeom.Mesh):
            continue
            
        path_str = prim.GetPath().pathString
        
        # 排除地板、地毯
        if is_excluded_by_name(path_str):
            continue
            
        # 计算 BBox
        obs_bbox = get_prim_world_bbox(prim, bbox_cache)
        if obs_bbox is None or obs_bbox.IsEmpty():
            continue
            
        # 排除太小的物体（如笔、杯子），不把它们视为“致命碰撞”
        size = obs_bbox.GetSize()
        area_xy = size[0] * size[1]
        if area_xy < MIN_LARGE_OBJECT_AREA:
            continue
            
        all_obstacles.append((path_str, obs_bbox))
        
    print(f"[INFO] Found {len(all_obstacles)} potential obstacles (walls, furniture, etc.)")

    # ==========================================
    # 2. 房间 Y 范围估算
    # ==========================================
    room_y_samples = []
    if ENABLE_ROOM_Y_FILTER:
        for prim in stage.Traverse():
            if not prim.GetName().startswith("Actor_"):
                continue
            info = is_candidate_actor_surface(prim, bbox_cache)
            if info:
                (cx, cy, z_top), _, _ = info
                room_y_samples.append(cy)

    if ENABLE_ROOM_Y_FILTER and room_y_samples:
        room_center_y = sum(room_y_samples) / len(room_y_samples)
        room_min_y = room_center_y - ROOM_Y_MARGIN
        room_max_y = room_center_y + ROOM_Y_MARGIN
        print(f"[INFO] Estimated room y-range: [{room_min_y:.3f}, {room_max_y:.3f}]")
    else:
        room_min_y = room_max_y = None

    # ==========================================
    # 3. 生成 Workspace
    # ==========================================
    workspace_count = 0

    for prim in stage.Traverse():
        name = prim.GetName()
        if not name.startswith("Actor_"):
            continue

        info = is_candidate_actor_surface(prim, bbox_cache)
        if info is None:
            continue

        (cx, cy, z_top), (sx, sy), (min_pt, max_pt, sz) = info

        # Room Filter
        if ENABLE_ROOM_Y_FILTER and room_min_y is not None:
            if not (room_min_y <= cy <= room_max_y):
                continue

        # 准备计算各个 space
        actor_path_str = prim.GetPath().pathString
        
        # --- 尺寸计算 ---
        fx = sx * FUNCTION_SCALE
        fy = sy * FUNCTION_SCALE
        dx = sx * DISTURB_SCALE
        dy = sy * DISTURB_SCALE

        if fx < MIN_SUBSPACE_SIZE or fy < MIN_SUBSPACE_SIZE: continue
        if dx < MIN_SUBSPACE_SIZE or dy < MIN_SUBSPACE_SIZE: continue

        table_min_y = min_pt[1]
        table_max_y = max_pt[1]
        table_min_x = min_pt[0]
        table_max_x = max_pt[0]

        # ==============================================
        # A. 计算 FunctionSpace & 碰撞检测
        # ==============================================
        front_margin = 0.10
        # 确保不超出桌子物理范围
        avail_fy = (table_max_y - table_min_y) - 2 * front_margin
        final_fy = min(fy, avail_fy)
        
        if final_fy < MIN_SUBSPACE_SIZE:
            continue

        func_center_y = table_min_y + front_margin + final_fy * 0.5
        func_bottom_z = z_top + PLANE_EPS_Z
        func_center_z = func_bottom_z + FUNCTION_THICKNESS * 0.5
        func_center = (cx, func_center_y, func_center_z)
        func_size = (fx, final_fy, FUNCTION_THICKNESS)

        # check collision
        if check_collision_with_obstacles(func_center, func_size, all_obstacles, actor_path_str):
            print(f"      -> Skip {name}: FunctionSpace collides with other objects.")
            continue

        # ==============================================
        # B. 计算 DisturbSpace & 碰撞检测
        # ==============================================
        back_margin = 0.10
        avail_dy = (table_max_y - table_min_y) - 2 * back_margin
        final_dy = min(dy, avail_dy)

        if final_dy < MIN_SUBSPACE_SIZE:
            continue

        dist_center_y = table_max_y - back_margin - final_dy * 0.5
        dist_bottom_z = z_top + PLANE_EPS_Z
        dist_center_z = dist_bottom_z + DISTURB_THICKNESS * 0.5
        dist_center = (cx, dist_center_y, dist_center_z)
        dist_size = (dx, final_dy, DISTURB_THICKNESS)

        # check collision
        if check_collision_with_obstacles(dist_center, dist_size, all_obstacles, actor_path_str):
            print(f"      -> Skip {name}: DisturbSpace collides with other objects.")
            continue

        # ==============================================
        # C. RobotLocation 自适应逻辑 (X优先，Y兜底)
        # ==============================================
        robot_size = (ROBOT_LOC_SIZE_X, ROBOT_LOC_SIZE_Y, ROBOT_THICKNESS)
        final_robot_center = None
        
        # --- Option 1: X Offset (Side) ---
        # 尝试放在右侧 (X+)
        # 坐标: cx + (桌宽/2) + 偏移
        robot_x_pos = table_max_x + ROBOT_OFFSET_DISTANCE
        robot_center_x_opt = (robot_x_pos + ROBOT_LOC_SIZE_X * 0.5, cy, ROBOT_THICKNESS * 0.5)
        
        # 检查 X 方案碰撞
        is_x_collided = check_collision_with_obstacles(
            robot_center_x_opt, robot_size, all_obstacles, actor_path_str
        )

        if not is_x_collided:
            final_robot_center = robot_center_x_opt
            # print(f"      -> {name}: Robot selected X-offset (Side)")
        else:
            # --- Option 2: Y Offset (Front) ---
            # 尝试放在前方 (Y-)
            # 坐标: cy - (桌深/2) - 偏移
            robot_y_pos = table_min_y - ROBOT_OFFSET_DISTANCE
            robot_center_y_opt = (cx, robot_y_pos - ROBOT_LOC_SIZE_Y * 0.5, ROBOT_THICKNESS * 0.5)
            
            # 检查 Y 方案碰撞
            is_y_collided = check_collision_with_obstacles(
                robot_center_y_opt, robot_size, all_obstacles, actor_path_str
            )
            
            if not is_y_collided:
                final_robot_center = robot_center_y_opt
                # print(f"      -> {name}: Robot selected Y-offset (Front) due to X collision")
            else:
                print(f"      -> Skip {name}: Robot collides in both X and Y positions.")
                continue

        # ==============================================
        # D. 保存结果
        # ==============================================
        workspace_count += 1
        workspace_name = f"workspace_{workspace_count:03d}.usd"
        workspace_path = os.path.join(workspace_dir, workspace_name)

        ws_stage = Usd.Stage.CreateNew(workspace_path)
        ws_stage.SetDefaultPrim(ws_stage.DefinePrim("/Workspace", "Xform"))

        create_colored_box_mesh(
            ws_stage, "/Workspace/FunctionSpace", func_center, func_size, (1.0, 0.0, 0.0)
        )
        create_colored_box_mesh(
            ws_stage, "/Workspace/DisturbSpace", dist_center, dist_size, (0.0, 1.0, 0.0)
        )
        create_colored_box_mesh(
            ws_stage, "/Workspace/RobotLocation", final_robot_center, robot_size, (0.0, 0.0, 1.0)
        )

        ws_stage.GetRootLayer().Save()
        print(f"      -> Saved: {workspace_path} (Robot: {'X-Side' if final_robot_center == robot_center_x_opt else 'Y-Front'})")

    print(f"[DONE] Generated {workspace_count} workspace files.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_workspace_annotator.py /path/to/scene.usd")
        sys.exit(1)

    scene_path = sys.argv[1]
    generate_workspaces(scene_path)