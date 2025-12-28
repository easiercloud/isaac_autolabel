import os
import sys
import math
import time
import json
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf

# ================== 全局单位控制 ==================
# 输入场景单位转换 (cm -> m)
SCENE_UNIT_SCALE = 0.01 

# 输出单位控制 (1.0 = 米, 100.0 = 厘米)
# 您要求输出为 cm，所以这里设为 100.0
OUTPUT_UNIT_SCALE = 100.0 

# ================== 核心参数 (内部逻辑统一使用 米制) ==================

# 1. 尺寸参数 (基于 m)
MIN_WORKSPACE_WIDTH = 0.30
MIN_WORKSPACE_DEPTH = 0.30
MIN_Z_TOP = 0.15
MAX_Z_TOP = 1.60  # 对应 160cm

# 2. 空间生成参数
FUNCTION_SCALE = 0.6
FUNCTION_THICKNESS = 0.5
DISTURB_SCALE = 0.7  # Disturb Space 稍大一点
DISTURB_THICKNESS = 0.5
PLANE_EPS_Z = 0.01   # 抬高 2cm，避开桌面微小起伏

# 3. 机器人参数
ROBOT_LOC_SIZE_X = 0.30
ROBOT_LOC_SIZE_Y = 0.30
ROBOT_THICKNESS  = 1
ROBOT_OFFSET_DISTANCE = 0.15

# 4. 碰撞检测参数
MIN_LARGE_OBJECT_AREA = 0.08 * 0.08  # 过滤小于 8cm 的物体 (基于米制计算)
COLLISION_RATIO_THRESHOLD = 0.05     # 允许 5% 的体积重叠
MIN_IGNORE_COLLISION_AREA = 0.05 * 0.05

# ================== 路径与分类映射 ==================

SEARCH_ROOTS = ["/Root/Meshes/Furnitures", "/Root/Meshes/Animation"]

ALLOWED_KEYWORDS = [
    "table", "desk", "counter", "island", "bar",
    "cabinet", "shelf", "cupboard", "drawer", "stand"
]

CATEGORY_MAPPING = {
    "table":   ["table", "desk", "counter", "island", "bar"],
    "cabinet": ["cabinet", "drawer", "shelf", "cupboard", "stand"],
    "carpet":  ["carpet", "rug", "mat"],
    "chair":   ["chair", "stool", "seat", "sofa", "couch"],
    "sofa":    ["sofa", "couch"],
    "wall":    ["wall"],
    "floor":   ["floor", "ground"],
    "ceiling": ["ceiling"]
}

# ================== 基础工具函数 ==================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_prim_world_bbox_scaled(prim, bbox_cache):
    """
    获取 BBox 并立即应用单位转换 (cm -> m)
    """
    bbox_cache.SetTime(Usd.TimeCode.Default())
    world_bound = bbox_cache.ComputeWorldBound(prim)
    box = world_bound.GetBox()
    
    if box.IsEmpty():
        return None
    
    # 获取原始数值 (cm)
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    
    # 转换为米 (m)
    min_pt_m = min_pt * SCENE_UNIT_SCALE
    max_pt_m = max_pt * SCENE_UNIT_SCALE
    
    return Gf.Range3d(min_pt_m, max_pt_m)

def rect_size_from_bbox(box):
    min_pt, max_pt = box.GetMin(), box.GetMax()
    size = box.GetSize()
    cx, cy = (min_pt[0] + max_pt[0]) * 0.5, (min_pt[1] + max_pt[1]) * 0.5
    return (cx, cy, max_pt[2]), (size[0], size[1]), (min_pt, max_pt, size[2])

def calculate_robot_quaternion(robot_pos, look_at_pos):
    """计算机器人朝向 (Z轴旋转)"""
    yaw = math.atan2(look_at_pos[1] - robot_pos[1], look_at_pos[0] - robot_pos[0])
    half = yaw * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]

def create_colored_box_mesh(stage, path, center, size, color):
    """创建可视化 Box (Debug USD 用)"""
    # 注意：这里的 center 和 size 已经是经过 OUTPUT_UNIT_SCALE 转换后的数值
    mesh = UsdGeom.Mesh.Define(stage, path)
    sx, sy, sz = size
    hx, hy, hz = sx/2, sy/2, sz/2
    points = [Gf.Vec3f(-hx,-hy,-hz), Gf.Vec3f(hx,-hy,-hz), Gf.Vec3f(hx,hy,-hz), Gf.Vec3f(-hx,hy,-hz),
              Gf.Vec3f(-hx,-hy,hz), Gf.Vec3f(hx,-hy,hz), Gf.Vec3f(hx,hy,hz), Gf.Vec3f(-hx,hy,hz)]
    indices = [0,1,2, 0,2,3, 4,6,5, 4,7,6, 3,2,6, 3,6,7, 0,5,1, 0,4,5, 0,3,7, 0,7,4, 1,5,6, 1,6,2]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateFaceVertexCountsAttr([3]*12)
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(center))

# ================== 碰撞检测核心逻辑 ==================

def get_collision_ratio(target_box, obstacles, self_prim_path):
    # [修复] 增加体积计算的健壮性
    size = target_box.GetSize()
    target_vol = size[0] * size[1] * size[2]
    
    # 如果体积太小或为负(无效盒子)，直接返回 1.0 (表示完全碰撞/不可用)，避免报错
    if target_vol <= 1e-6: 
        return 1.0, "InvalidTargetBox"

    max_vol = 0.0
    hit_name = None
    
    # 获取父路径字符串 (例如 /Root/Meshes/Furnitures/teatable)
    # 用于排除同组的所有部件 (例如桌腿、抽屉等)
    self_parent_path = self_prim_path.rsplit('/', 1)[0]
    
    for obs_path, obs_box in obstacles:
        # [关键修复] 只要障碍物属于同一个父组，就忽略 (视为物体自身的一部分)
        if obs_path.startswith(self_parent_path): 
            continue
        
        intersect = Gf.Range3d(target_box)
        intersect.IntersectWith(obs_box)
        if not intersect.IsEmpty():
            sz = intersect.GetSize()
            vol = sz[0] * sz[1] * sz[2]
            if vol > max_vol:
                max_vol = vol
                hit_name = obs_path
                
    return max_vol / target_vol, hit_name

def find_best_space_on_surface(surface_info, obstacles, prim_path):
    (cx, cy, z_top), (sx, sy) = surface_info
    
    # 策略：网格搜索 (Grid Search)
    # 尝试在桌面上寻找空闲区域
    
    # 1. 定义要尝试的缩放比例 (先试大的，再试小的)
    check_scales = [0.6, 0.4] 
    
    for scale in check_scales:
        test_sx, test_sy = sx * scale, sy * scale
        
        # 计算可以移动的余量
        margin_x = (sx - test_sx) / 2.0
        margin_y = (sy - test_sy) / 2.0
        
        # 定义采样点：中心 + 上下左右偏移
        offsets = [
            (0,0), 
            (margin_x, 0), (-margin_x, 0), (0, margin_y), (0, -margin_y)
        ]
        
        for ox, oy in offsets:
            tx, ty = cx + ox, cy + oy
            # 功能区稍微悬浮在桌面上方
            tz_center = z_top + PLANE_EPS_Z + FUNCTION_THICKNESS/2
            
            t_box = Gf.Range3d(
                Gf.Vec3d(tx - test_sx/2, ty - test_sy/2, z_top + PLANE_EPS_Z),
                Gf.Vec3d(tx + test_sx/2, ty + test_sy/2, z_top + PLANE_EPS_Z + FUNCTION_THICKNESS)
            )
            
            ratio, hit = get_collision_ratio(t_box, obstacles, prim_path)
            
            # 如果碰撞比例极小，则认为找到了可用空间
            if ratio < COLLISION_RATIO_THRESHOLD:
                return (tx, ty, tz_center), (test_sx, test_sy, FUNCTION_THICKNESS), True, "OK"
                
    return None, None, False, "Blocked by obstacles"

def find_robot_spot(surface_info, obstacles, prim_path):
    (cx, cy, z_top), (sx, sy) = surface_info
    
    # 搜索半径：桌面半径 + 偏移量 + 机器人自身半径
    radius_x = sx/2 + ROBOT_OFFSET_DISTANCE + ROBOT_LOC_SIZE_X/2
    radius_y = sy/2 + ROBOT_OFFSET_DISTANCE + ROBOT_LOC_SIZE_Y/2
    
    # 策略：沿桌子边缘 360 度搜索 (24个采样点)
    for i in range(24): 
        angle = (i * 15) * (math.pi / 180.0)
        tx = cx + radius_x * math.cos(angle)
        ty = cy + radius_y * math.sin(angle)
        
        # [修复] Z轴范围：从离地 5cm (0.05) 检测到离地 25cm (0.25)
        # 这样既避开了地毯(通常<2cm)，又能检测到周围的家具腿或墙壁
        t_box = Gf.Range3d(
            Gf.Vec3d(tx - ROBOT_LOC_SIZE_X/2, ty - ROBOT_LOC_SIZE_Y/2, 0.05),
            Gf.Vec3d(tx + ROBOT_LOC_SIZE_X/2, ty + ROBOT_LOC_SIZE_Y/2, 0.25)
        )
        
        ratio, _ = get_collision_ratio(t_box, obstacles, prim_path)
        
        # 允许少量碰撞 (例如地毯或稍微碰到墙角)
        if ratio < 0.1: 
            return (tx, ty, ROBOT_THICKNESS/2)
            
    return None

# ================== 场景信息收集 ==================

def collect_scene_info(stage):
    """收集灯光和所有物体的分类信息 (JSON 输出用)"""
    lights = {"Dome": [], "Distant": [], "Rect": []}
    material_obj_prim = {k: [] for k in CATEGORY_MAPPING.keys()}
    
    for prim in stage.Traverse():
        path_str = prim.GetPath().pathString
        name_lower = prim.GetName().lower()
        
        # 收集灯光
        if prim.IsA(UsdLux.DomeLight): lights["Dome"].append(path_str)
        elif prim.IsA(UsdLux.DistantLight): lights["Distant"].append(path_str)
        elif prim.IsA(UsdLux.RectLight): lights["Rect"].append(path_str)
        
        # 收集材质物体 (基于名称关键词)
        # 注意：这里我们收集的是所有 Mesh，用于 JSON 记录，不做筛选
        if prim.IsA(UsdGeom.Mesh):
            for category, keywords in CATEGORY_MAPPING.items():
                for kw in keywords:
                    if kw in name_lower or kw in path_str.lower():
                        if path_str not in material_obj_prim[category]:
                            material_obj_prim[category].append(path_str)
                        break 
                        
    return lights, material_obj_prim

# ================== 主程序 ==================

def generate_workspaces(scene_usd_path):
    start_time = time.time()
    
    stage = Usd.Stage.Open(scene_usd_path)
    if not stage:
        print(f"[ERROR] Cannot open stage: {scene_usd_path}")
        return

    print(f"[INFO] Processing: {scene_usd_path}")
    print(f"[INFO] Output Unit Scale: {OUTPUT_UNIT_SCALE} (1.0=m, 100.0=cm)")
    
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    
    # 1. 收集障碍物 (并转换单位 cm -> m)
    obstacles = []
    skipped_count = 0
    
    for root in SEARCH_ROOTS:
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim.IsValid(): continue
        
        for cat in root_prim.GetChildren():
            for model in cat.GetChildren():
                # 识别 GRScenes 的 model_xxx 结构
                if model.GetName().startswith("model_") or model.GetName().startswith("SM_"):
                    # [重点] 使用 _scaled 函数，获取米制 BBox
                    bb = get_prim_world_bbox_scaled(model, bbox_cache)
                    if bb and not bb.IsEmpty():
                        sz = bb.GetSize()
                        # 过滤太小的物体
                        if (sz[0]*sz[1]) > MIN_LARGE_OBJECT_AREA:
                            obstacles.append((model.GetPath().pathString, bb))
                        else:
                            skipped_count += 1
    
    print(f"[INFO] Obstacles found: {len(obstacles)} (Skipped {skipped_count} small objects)")

    # 2. 收集场景信息
    scene_lights, scene_materials = collect_scene_info(stage)
    
    # 3. 处理候选物体并生成 Workspace
    json_data = {
        "function_space_objects": {},
        "robot_init_pose": {},
        "lights": scene_lights
    }
    
    ws_count = 0
    workspace_dir = os.path.join(os.path.dirname(scene_usd_path), "workspace_meta")
    ensure_dir(workspace_dir)

    for root in SEARCH_ROOTS:
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim.IsValid(): continue
        
        for cat in root_prim.GetChildren():
            cat_name = cat.GetName().lower()
            
            # 白名单检查
            if not any(kw in cat_name for kw in ALLOWED_KEYWORDS): continue
            
            for model in cat.GetChildren():
                if not (model.GetName().startswith("model_") or model.GetName().startswith("SM_")): continue
                
                # 获取米制 BBox
                bb = get_prim_world_bbox_scaled(model, bbox_cache)
                if not bb: continue
                (cx, cy, z_top), (sx, sy), _ = rect_size_from_bbox(bb)
                
                # 尺寸与高度过滤 (米制标准)
                if sx < MIN_WORKSPACE_WIDTH or sy < MIN_WORKSPACE_DEPTH: continue
                if not (MIN_Z_TOP <= z_top <= MAX_Z_TOP): continue

                prim_path = model.GetPath().pathString
                surface_info = ((cx, cy, z_top), (sx, sy))
                
                # A. 寻找功能区 (Function Space)
                fc, fs, f_ok, f_msg = find_best_space_on_surface(surface_info, obstacles, prim_path)
                
                if not f_ok:
                    print(f"  [FAIL] {model.GetName()}: {f_msg}")
                    continue
                
                # B. 寻找机器人位置 (Robot Spot)
                robot_spot = find_robot_spot(surface_info, obstacles, prim_path)
                
                if not robot_spot:
                    print(f"  [FAIL] {model.GetName()}: Robot location blocked")
                    continue
                
                # --- 数据后处理：应用输出单位 (m -> cm) ---
                # 将所有计算出的米制坐标乘以 OUTPUT_UNIT_SCALE
                
                fc_out = [x * OUTPUT_UNIT_SCALE for x in fc]
                fs_out = [x * OUTPUT_UNIT_SCALE for x in fs]
                
                robot_spot_out = [x * OUTPUT_UNIT_SCALE for x in robot_spot]
                
                # Disturb Space (使用同样的逻辑)
                dx, dy = sx * DISTURB_SCALE, sy * DISTURB_SCALE
                ds = (dx, dy, DISTURB_THICKNESS)
                dc = (fc[0], fc[1], fc[2] + (DISTURB_THICKNESS - FUNCTION_THICKNESS)/2) 
                
                dc_out = [x * OUTPUT_UNIT_SCALE for x in dc]
                ds_out = [x * OUTPUT_UNIT_SCALE for x in ds]
                
                # C. 生成数据
                ws_name = f"workspace_{ws_count:02d}"

                # 填充 JSON
                json_data["function_space_objects"][ws_name] = {
                    "position": fc_out, 
                    "quaternion": [1.0, 0.0, 0.0, 0.0], 
                    "size": fs_out,
                    "material_obj_prim": scene_materials
                }
                json_data["function_space_objects"][f"{ws_name}_extra"] = {
                    "position": dc_out, 
                    "quaternion": [1.0, 0.0, 0.0, 0.0], 
                    "size": ds_out
                }
                json_data["robot_init_pose"][ws_name] = {
                    "position": robot_spot_out,
                    "quaternion": calculate_robot_quaternion(robot_spot, (cx, cy))
                }
                
                # 生成 Debug USD (确保 Mesh 也使用转换后的单位)
                usd_path = os.path.join(workspace_dir, f"{ws_name}.usd")
                new_stage = Usd.Stage.CreateNew(usd_path)
                new_stage.SetDefaultPrim(new_stage.DefinePrim("/Workspace", "Xform"))
                
                create_colored_box_mesh(new_stage, "/Workspace/FunctionSpace", fc_out, fs_out, (1, 0, 0))
                create_colored_box_mesh(new_stage, "/Workspace/DisturbSpace", dc_out, ds_out, (0, 1, 0))
                # 机器人尺寸也要转换
                robot_size_out = [ROBOT_LOC_SIZE_X * OUTPUT_UNIT_SCALE, ROBOT_LOC_SIZE_Y * OUTPUT_UNIT_SCALE, ROBOT_THICKNESS * OUTPUT_UNIT_SCALE]
                create_colored_box_mesh(new_stage, "/Workspace/Robot", robot_spot_out, robot_size_out, (0, 0, 1))
                
                new_stage.GetRootLayer().Save()
                
                print(f"  [GEN] {ws_name} -> {model.GetName()}")
                ws_count += 1

    # 写出 JSON 文件
    json_path = os.path.join(os.path.dirname(scene_usd_path), "scene_parameters.json")
    try:
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"[INFO] Saved scene parameters to: {json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {e}")
        
    print(f"[DONE] Generated {ws_count} workspaces.")
    print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_workspace_annotator.py /path/to/scene.usd")
        sys.exit(1)
    generate_workspaces(sys.argv[1])