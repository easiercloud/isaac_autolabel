import os
import sys
from pxr import Usd, UsdGeom, Gf

def inspect_usd_file(usd_path):
    if not os.path.exists(usd_path):
        print(f"[Error] File not found: {usd_path}")
        return

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[Error] Could not open stage: {usd_path}")
        return

    print(f"=== Inspecting: {usd_path} ===\n")

    # 初始化包围盒计算缓存
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    )

    # 我们关注的根目录
    target_roots = [
        "/Root/Meshes/Furnitures",
        "/Root/Meshes/Animation"
    ]

    found_count = 0

    # 遍历整个 Stage
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        
        # 只检查我们关心的目录
        is_target = False
        for root in target_roots:
            if path.startswith(root):
                is_target = True
                break
        
        if not is_target:
            continue

        name = prim.GetName()
        
        # 这里的过滤逻辑：我们想看 'model_' 开头的节点，或者是其第一层子节点
        # 这样可以看出究竟是父节点有位置信息，还是子节点有位置信息
        # 同时排除一些细碎的组件(Component_)以减少日志量，除非它们是 Mesh
        is_model = "model_" in name or "SM_" in name
        is_group = "Group_" in name
        
        if not (is_model or is_group):
            continue

        found_count += 1
        
        # --- 1. 获取基础属性 ---
        prim_type = prim.GetTypeName()
        
        # --- 2. 计算世界坐标变换 (World Transform) ---
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = world_transform.ExtractTranslation()
            # rotation = world_transform.ExtractRotation() # 暂时不需要旋转细节
            scale = world_transform.ExtractRotationMatrix().GetRow(0).GetLength() # 粗略估算缩放
        else:
            translation = Gf.Vec3d(0,0,0)
            scale = 1.0

        # --- 3. 计算世界包围盒 (World Bounding Box) ---
        # 这一步至关重要，看是否能正确算出体积
        bbox = bbox_cache.ComputeWorldBound(prim).GetBox()
        
        if bbox.IsEmpty():
            size = Gf.Vec3d(0,0,0)
            center = Gf.Vec3d(0,0,0)
            z_min, z_max = 0, 0
        else:
            size = bbox.GetSize()
            center = (bbox.GetMin() + bbox.GetMax()) * 0.5
            z_min = bbox.GetMin()[2]
            z_max = bbox.GetMax()[2]

        # --- 4. 打印信息 ---
        print("-" * 60)
        print(f"PATH:   {path}")
        print(f"TYPE:   {prim_type}")
        print(f"WORLD POS (Origin):   x={translation[0]:.4f}, y={translation[1]:.4f}, z={translation[2]:.4f}")
        print(f"BBOX CENTER:          x={center[0]:.4f}, y={center[1]:.4f}, z={center[2]:.4f}")
        print(f"BBOX SIZE (W/D/H):    x={size[0]:.4f}, y={size[1]:.4f}, z={size[2]:.4f}")
        print(f"Z RANGE (Height):     [{z_min:.4f} ~ {z_max:.4f}]")
        
        # 简单的分析提示
        analysis = []
        if size[0] < 0.01 or size[1] < 0.01:
            analysis.append(">> [WARN] Size too small (Micro object?)")
        if z_max < 0.1:
            analysis.append(">> [WARN] Too low (On floor or flat?)")
        if z_max > 2.0:
            analysis.append(">> [INFO] Tall object")
        
        if analysis:
            print("ANALYSIS: " + " | ".join(analysis))

    if found_count == 0:
        print("\n[WARNING] No objects found matching 'model_' or 'Group_' in target roots.")
        print("Please verify the target_roots path in the script matches your USD structure.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_usd_structure.py /path/to/file.usd")
    else:
        inspect_usd_file(sys.argv[1])