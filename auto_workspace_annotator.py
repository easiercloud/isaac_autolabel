import os
import sys
import math
import time  # 新增：用于计时

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

# ----------------- 可调参数 -----------------

# 最小可操作平面尺寸（单位：米）
MIN_WORKSPACE_WIDTH = 0.40   # 40 cm
MIN_WORKSPACE_DEPTH = 0.40   # 40 cm

# function_space / disturb_space 尺寸相对于平面 bbox 的比例
FUNCTION_SCALE = 0.7
DISTURB_SCALE = 0.8

# function_space / disturb_space 的最小尺寸（单位：米）
MIN_SUBSPACE_SIZE = 0.05  # 5 cm

# Robot base footprint 尺寸
ROBOT_LOC_SIZE_X = 0.40   # 40 cm
ROBOT_LOC_SIZE_Y = 0.40   # 40 cm
ROBOT_THICKNESS  = 0.03   # 1 cm

# Robot 相对 workspace 的偏移距离（单位米）
ROBOT_OFFSET_DISTANCE = 0  # 15 cm

# 厚度阈值，用于过滤墙面等（Z 方向 thickness）
MAX_PLANE_THICKNESS = 2.0  # 2m

# 可操作平面高度范围（台面高度）
MIN_Z_TOP = 0.6   # 0.6 m
MAX_Z_TOP = 1.2   # 1.2 m

# 工作区块厚度（cube 的高度），单位米
FUNCTION_THICKNESS = 0.03   # 3 cm
DISTURB_THICKNESS  = 0.03   # 3 cm

# 放在平面上方的小抬升，避免 z fighting
PLANE_EPS_Z = 0.005  # 0.5 cm

# 要排除的名字关键字
EXCLUDE_NAME_KEYWORDS = [
    "carpet", "rug", "floor", "ground", "wall", "ceiling", "sofa", "chair"
]

# 认为是“大物体”的最小投影面积（XY）
MIN_LARGE_OBJECT_AREA = 0.05 * 0.05    # 0.0025 m^2

# 碰撞判定参数
COLLISION_RATIO_THRESHOLD = 1  # 严重碰撞阈值 (30%)
SHRINK_STEP = 0.05              # 收缩步长 (5 cm),收缩次数20次
MIN_IGNORE_COLLISION_AREA = 0.05 * 0.05 # 忽略不计的微小碰撞面积 (5cm*5cm)

ENABLE_ROOM_Y_FILTER = True
ROOM_Y_MARGIN = 15 # 修改：稍微放大一点容差

# ----------------- 基础工具 -----------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def is_excluded_by_name(path_string):
    s = path_string.lower()
    return any(kw in s for kw in EXCLUDE_NAME_KEYWORDS)

def get_prim_world_bbox(prim, bbox_cache):
    local_bound = bbox_cache.ComputeLocalBound(prim)
    local_box = local_bound.GetBox()
    if local_box.IsEmpty():
        return None
    xformable = UsdGeom.Xformable(prim)
    local_to_world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_bbox_calc = Gf.BBox3d(local_box, local_to_world_transform)
    return world_bbox_calc.ComputeAlignedRange()

def rect_size_from_bbox(box: Gf.Range3d):
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    size = box.GetSize()
    center = (min_pt + max_pt) * 0.5
    cx, cy, _ = center[0], center[1], center[2]
    sx, sy, sz = size[0], size[1], size[2]
    z_top = max_pt[2]
    return (cx, cy, z_top), (sx, sy), (min_pt, max_pt, sz)

def create_colored_box_mesh(stage, path, center, size_xyz, color, z_offset=0.0):
    sx, sy, sz = size_xyz
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [
        Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f( hx, -hy, -hz),
        Gf.Vec3f( hx,  hy, -hz), Gf.Vec3f(-hx,  hy, -hz),
        Gf.Vec3f(-hx, -hy,  hz), Gf.Vec3f( hx, -hy,  hz),
        Gf.Vec3f( hx,  hy,  hz), Gf.Vec3f(-hx,  hy,  hz),
    ]
    face_vertex_indices = [
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        3, 2, 6, 3, 6, 7, 0, 5, 1, 0, 4, 5,
        0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
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

# ----------------- 碰撞检测核心逻辑 -----------------

def get_collision_info(target_box: Gf.Range3d, obstacles, self_actor_path):
    target_volume = target_box.GetSize()[0] * target_box.GetSize()[1] * target_box.GetSize()[2]
    if target_volume <= 1e-6:
        return 0.0, Gf.Range3d(), None

    max_vol = 0.0
    max_intersect_box = Gf.Range3d()
    hit_name = None

    for obs_path, obs_box in obstacles:
        if obs_path.startswith(self_actor_path):
            continue
            
        intersection = Gf.Range3d(target_box)
        intersection.IntersectWith(obs_box)
        
        if not intersection.IsEmpty():
            sz = intersection.GetSize()
            vol = sz[0] * sz[1] * sz[2]
            if vol > max_vol:
                max_vol = vol
                max_intersect_box = intersection
                hit_name = obs_path

    ratio = max_vol / target_volume
    return ratio, max_intersect_box, hit_name

def optimize_space_box(center, size, obstacles, self_actor_path, space_type_name="Space"):
    curr_cx, curr_cy, curr_cz = center
    curr_sx, curr_sy, curr_sz = size
    
    curr_min_x = curr_cx - curr_sx * 0.5
    curr_max_x = curr_cx + curr_sx * 0.5
    curr_min_y = curr_cy - curr_sy * 0.5
    curr_max_y = curr_cy + curr_sy * 0.5
    
    iteration = 0
    max_iter = 20
    last_hit_name = "None"
    
    while iteration < max_iter:
        iteration += 1
        
        min_pt = Gf.Vec3d(curr_min_x, curr_min_y, curr_cz - curr_sz * 0.5)
        max_pt = Gf.Vec3d(curr_max_x, curr_max_y, curr_cz + curr_sz * 0.5)
        curr_box = Gf.Range3d(min_pt, max_pt)
        curr_sx = curr_max_x - curr_min_x
        curr_sy = curr_max_y - curr_min_y
        
        if curr_sx < MIN_SUBSPACE_SIZE or curr_sy < MIN_SUBSPACE_SIZE:
            return None, None, f"Shrunk too small ({curr_sx:.2f}x{curr_sy:.2f})", False

        ratio, hit_box, hit_name = get_collision_info(curr_box, obstacles, self_actor_path)
        
        hit_size = hit_box.GetSize()
        hit_area = hit_size[0] * hit_size[1]
        
        # Case A: Success
        if ratio == 0.0 or hit_area < MIN_IGNORE_COLLISION_AREA:
            final_center = ((curr_min_x + curr_max_x)*0.5, (curr_min_y + curr_max_y)*0.5, curr_cz)
            final_size = (curr_sx, curr_sy, curr_sz)
            if iteration > 1:
                return final_center, final_size, f"Optimized (Iter {iteration}): Shrunk to avoid {last_hit_name}", True
            else:
                return final_center, final_size, "Clean", True

        # Case B: Severe Collision
        if ratio > COLLISION_RATIO_THRESHOLD and iteration == 1:
            return None, None, f"Severe Collision ({ratio:.1%}) with {hit_name}", False
        
        # Case C: Shrink
        last_hit_name = hit_name
        hit_center = (hit_box.GetMin() + hit_box.GetMax()) * 0.5
        box_center_x = (curr_min_x + curr_max_x) * 0.5
        box_center_y = (curr_min_y + curr_max_y) * 0.5
        
        diff_x = hit_center[0] - box_center_x
        diff_y = hit_center[1] - box_center_y
        
        if abs(diff_x) * curr_sy > abs(diff_y) * curr_sx: 
            if diff_x > 0: curr_max_x -= SHRINK_STEP
            else:          curr_min_x += SHRINK_STEP
        else:
            if diff_y > 0: curr_max_y -= SHRINK_STEP
            else:          curr_min_y += SHRINK_STEP
            
    return None, None, "Max Iterations Reached", False

# ----------------- Actor 分析逻辑 -----------------

def list_meshes_under_actor(actor_prim):
    meshes = []
    for child in Usd.PrimRange(actor_prim):
        if child == actor_prim: continue
        if child.IsA(UsdGeom.Mesh):
            meshes.append(child)
    return meshes

def get_actor_surface_info(actor_prim, bbox_cache):
    meshes = list_meshes_under_actor(actor_prim)
    if not meshes: return None
    world_bboxes = []
    for mesh_prim in meshes:
        wb = get_prim_world_bbox(mesh_prim, bbox_cache)
        if wb and not wb.IsEmpty():
            world_bboxes.append(wb)
    if not world_bboxes: return None
    overall_box = Gf.Range3d()
    for box in world_bboxes:
        overall_box.UnionWith(box)
    if overall_box.IsEmpty(): return None
    return rect_size_from_bbox(overall_box)

def check_candidate_actor_surface(actor_prim, bbox_cache):
    if is_excluded_by_name(actor_prim.GetPath().pathString):
        return False, "Excluded by name (keyword)", None

    info = get_actor_surface_info(actor_prim, bbox_cache)
    if info is None:
        return False, "No valid mesh/bbox found", None

    (cx, cy, z_top), (sx, sy), (_, _, sz) = info

    if sz > MAX_PLANE_THICKNESS:
        return False, f"Too thick (Z={sz:.2f}m > {MAX_PLANE_THICKNESS}m)", None

    cond1 = (sx >= MIN_WORKSPACE_WIDTH and sy >= MIN_WORKSPACE_DEPTH)
    cond2 = (sy >= MIN_WORKSPACE_WIDTH and sx >= MIN_WORKSPACE_DEPTH)
    if not (cond1 or cond2):
        return False, f"Too small (Size={sx:.2f}x{sy:.2f} < Min {MIN_WORKSPACE_WIDTH})", None

    if not (MIN_Z_TOP <= z_top <= MAX_Z_TOP):
        return False, f"Invalid height (Z_top={z_top:.2f}m not in [{MIN_Z_TOP}, {MAX_Z_TOP}])", None

    return True, "Valid", info

# ----------------- 主逻辑 -----------------

def generate_workspaces(scene_usd_path):
    start_time = time.time()
    
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

    # 1. Obstacles
    print("[INFO] Building obstacle list...")
    all_obstacles = [] 
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        path_str = prim.GetPath().pathString
        if is_excluded_by_name(path_str): continue
        obs_bbox = get_prim_world_bbox(prim, bbox_cache)
        if obs_bbox is None or obs_bbox.IsEmpty(): continue
        
        size = obs_bbox.GetSize()
        area_xy = size[0] * size[1]
        if area_xy < MIN_LARGE_OBJECT_AREA: continue
        all_obstacles.append((path_str, obs_bbox))
    print(f"[INFO] Found {len(all_obstacles)} potential obstacles.")

    # 2. Room Y Filter
    room_y_samples = []
    if ENABLE_ROOM_Y_FILTER:
        for prim in stage.Traverse():
            if not prim.GetName().startswith("Actor_"): continue
            success, reason, info = check_candidate_actor_surface(prim, bbox_cache)
            if success:
                (cx, cy, z_top), _, _ = info
                room_y_samples.append(cy)

    room_min_y = room_max_y = None
    if ENABLE_ROOM_Y_FILTER and room_y_samples:
        room_center_y = sum(room_y_samples) / len(room_y_samples)
        room_min_y = room_center_y - ROOM_Y_MARGIN
        room_max_y = room_center_y + ROOM_Y_MARGIN
        print(f"[INFO] Room Y-Filter Active: Mean={room_center_y:.2f}, Range=[{room_min_y:.2f}, {room_max_y:.2f}]")

    # 3. Processing
    workspace_count = 0
    
    for prim in stage.Traverse():
        name = prim.GetName()
        if not name.startswith("Actor_"): continue
        
        log_prefix = f"[{name}]"

        # Step 1: Surface Check
        is_valid, reason, info = check_candidate_actor_surface(prim, bbox_cache)
        if not is_valid:
            print(f"{log_prefix} Skipped: {reason}")
            continue

        (cx, cy, z_top), (sx, sy), (min_pt, max_pt, sz) = info

        # Step 2: Room Filter
        if ENABLE_ROOM_Y_FILTER and room_min_y is not None:
            if not (room_min_y <= cy <= room_max_y):
                print(f"{log_prefix} Skipped: Outside Room Y Range (Y={cy:.2f})")
                continue

        actor_path_str = prim.GetPath().pathString
        reasons = []

        # --- Initial Size ---
        fx = sx * FUNCTION_SCALE
        fy = sy * FUNCTION_SCALE
        dx = sx * DISTURB_SCALE
        dy = sy * DISTURB_SCALE

        if fx < MIN_SUBSPACE_SIZE or fy < MIN_SUBSPACE_SIZE: 
            print(f"{log_prefix} Skipped: Subspace init size too small ({fx:.2f}x{fy:.2f})")
            continue

        table_min_y = min_pt[1]
        table_max_y = max_pt[1]
        table_max_x = max_pt[0]
        
        # --- A. FunctionSpace ---
        front_margin = 0.10
        avail_fy = (table_max_y - table_min_y) - 2 * front_margin
        final_fy = min(fy, avail_fy)
        
        func_center_y = table_min_y + front_margin + final_fy * 0.5
        func_z = z_top + PLANE_EPS_Z + FUNCTION_THICKNESS * 0.5
        func_center_init = (cx, func_center_y, func_z)
        func_size_init = (fx, final_fy, FUNCTION_THICKNESS)
        
        fc, fs, f_status, f_ok = optimize_space_box(
            func_center_init, func_size_init, all_obstacles, actor_path_str, "FunctionSpace"
        )
        if not f_ok:
            print(f"{log_prefix} Skipped: FunctionSpace failed -> {f_status}")
            continue
        reasons.append(f"Func: {f_status}")

        # --- B. DisturbSpace ---
        back_margin = 0.10
        avail_dy = (table_max_y - table_min_y) - 2 * back_margin
        final_dy = min(dy, avail_dy)
        
        dist_center_y = table_max_y - back_margin - final_dy * 0.5
        dist_z = z_top + PLANE_EPS_Z + DISTURB_THICKNESS * 0.5
        dist_center_init = (cx, dist_center_y, dist_z)
        dist_size_init = (dx, final_dy, DISTURB_THICKNESS)

        dc, ds, d_status, d_ok = optimize_space_box(
            dist_center_init, dist_size_init, all_obstacles, actor_path_str, "DisturbSpace"
        )
        if not d_ok:
            print(f"{log_prefix} Skipped: DisturbSpace failed -> {d_status}")
            continue
        reasons.append(f"Disturb: {d_status}")

        # --- C. RobotLocation (Refined Search Logic) ---
        robot_size = (ROBOT_LOC_SIZE_X, ROBOT_LOC_SIZE_Y, ROBOT_THICKNESS)
        final_robot_center = None
        robot_choice = ""
        found_spot = False

        # 搜索偏移列表
        # 0: 原始位置
        # 正数: 尝试更远的位置 (更安全)
        # 负数: 尝试更近的位置 (更紧凑)
        search_offsets = [0.15, -0.15, -0.20, 0.20, -0.10, 0.10, -0.05, 0.05]

        # 1. Try X Axis (Side)
        # Base Position: Right Side of table
        base_x_pos = table_max_x + ROBOT_OFFSET_DISTANCE + ROBOT_LOC_SIZE_X * 0.5
        
        for offset in search_offsets:
            # X方向：正偏移 = 向右(更远)，负偏移 = 向左(更近)
            test_cx = base_x_pos + offset
            test_center = (test_cx, cy, ROBOT_THICKNESS * 0.5)
            
            test_min = Gf.Vec3d(test_cx - ROBOT_LOC_SIZE_X/2, cy - ROBOT_LOC_SIZE_Y/2, 0)
            test_max = Gf.Vec3d(test_cx + ROBOT_LOC_SIZE_X/2, cy + ROBOT_LOC_SIZE_Y/2, ROBOT_THICKNESS)
            
            r_ratio, _, _ = get_collision_info(Gf.Range3d(test_min, test_max), all_obstacles, actor_path_str)
            
            if r_ratio < 0.1: 
                final_robot_center = test_center
                off_str = f"+{offset:.2f}" if offset >= 0 else f"{offset:.2f}"
                robot_choice = f"Side(X) {off_str}"
                found_spot = True
                break
        
        # 2. Try Y Axis (Front) - Fallback with Search
        if not found_spot:
            # Base Position: Front of table (Y minus direction)
            base_y_pos = table_min_y - ROBOT_OFFSET_DISTANCE - ROBOT_LOC_SIZE_Y * 0.5
            
            for offset in search_offsets:
                # Y方向：前方是负Y轴。
                # 正偏移 = 向前(负Y方向, 更远) => 需要减去 offset
                # 负偏移 = 向后(正Y方向, 更近) => 减去负数 = 加上
                test_cy = base_y_pos - offset
                test_center = (cx, test_cy, ROBOT_THICKNESS * 0.5)
                
                test_min = Gf.Vec3d(cx - ROBOT_LOC_SIZE_X/2, test_cy - ROBOT_LOC_SIZE_Y/2, 0)
                test_max = Gf.Vec3d(cx + ROBOT_LOC_SIZE_X/2, test_cy + ROBOT_LOC_SIZE_Y/2, ROBOT_THICKNESS)
                
                r_ratio_y, _, _ = get_collision_info(Gf.Range3d(test_min, test_max), all_obstacles, actor_path_str)
                
                if r_ratio_y < 0.1:
                    final_robot_center = test_center
                    off_str = f"+{offset:.2f}" if offset >= 0 else f"{offset:.2f}"
                    robot_choice = f"Front(Y) {off_str}"
                    found_spot = True
                    break
        
        # Final Decision
        if not found_spot:
            print(f"{log_prefix} Skipped: Robot location blocked in all X and Y search positions.")
            continue
            
        reasons.append(f"Robot: {robot_choice}")

        # --- D. Save ---
        workspace_count += 1
        workspace_name = f"workspace_{workspace_count:03d}.usd"
        workspace_path = os.path.join(workspace_dir, workspace_name)

        ws_stage = Usd.Stage.CreateNew(workspace_path)
        ws_stage.SetDefaultPrim(ws_stage.DefinePrim("/Workspace", "Xform"))

        create_colored_box_mesh(ws_stage, "/Workspace/FunctionSpace", fc, fs, (1.0, 0.0, 0.0))
        create_colored_box_mesh(ws_stage, "/Workspace/DisturbSpace", dc, ds, (0.0, 1.0, 0.0))
        create_colored_box_mesh(ws_stage, "/Workspace/RobotLocation", final_robot_center, robot_size, (0.0, 0.0, 1.0))

        ws_stage.GetRootLayer().Save()
        
        print(f"  -> Generated {workspace_name}: " + " | ".join(reasons))

    end_time = time.time()
    total_time = end_time - start_time
    
    print("-" * 50)
    print(f"[DONE] Generated {workspace_count} workspace files.")
    print(f"[TIME] Total Execution Time: {total_time:.4f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_workspace_annotator.py /path/to/scene.usd")
        sys.exit(1)
    scene_path = sys.argv[1]
    generate_workspaces(scene_path)