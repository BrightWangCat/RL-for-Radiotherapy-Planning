import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# =================配置区域=================
CASE_PATH = "/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans/pt_241"
OUTPUT_DIR = "./output_step2_state1"
FULL_SHAPE = (128, 128, 128)
TOTAL_VOXELS = 128 * 128 * 128

# 论文复现的关键：定义每个器官的“目标剂量值” (相对于处方剂量 70Gy)
# Key 是文件名中包含的字符串，Value 是目标值 (Objective Value)
# 逻辑参考：PTV=1.0, 脑干<50Gy(0.71), 脊髓<45Gy(0.64), 腮腺<26Gy(0.37)
OBJECTIVE_MAPPING = {
    "PTV70": 1.0,
    "PTV63": 0.9,
    "PTV56": 0.8,
    "Brainstem": 0.7,
    "SpinalCord": 0.64,
    "RightParotid": 0.37,
    "LeftParotid": 0.37,
    "Esophagus": 0.6,
    "Larynx": 0.6,
    "Mandible": 1.0, # 有时下颌骨允许高剂量，暂定
}
# =========================================

def load_sparse_csv(csv_path, is_structure=False):
    """(复用 Step 1 的代码) 读取稀疏CSV"""
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path)
        indices = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(-1).astype(int)
        valid_mask = (indices >= 0) & (indices < TOTAL_VOXELS)
        indices = indices[valid_mask]
        
        dense_flat = np.zeros(TOTAL_VOXELS, dtype=np.float32)
        if is_structure:
            dense_flat[indices] = 1.0
        else:
            values = pd.to_numeric(df.iloc[:, -1], errors='coerce').fillna(0.0)
            values = values[valid_mask]
            dense_flat[indices] = values
        return dense_flat.reshape(FULL_SHAPE)
    except Exception as e:
        print(f"Error {csv_path}: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"Processing case: {CASE_PATH}")

    # ---------------------------------------------------------
    # 1. 重新生成 Normalized Dose (基于 Step 1 的成果)
    # ---------------------------------------------------------
    dose_path = os.path.join(CASE_PATH, "dose.csv")
    dose_grid = load_sparse_csv(dose_path, is_structure=False)
    
    # 找主 PTV (PTV70)
    ptv_files = glob.glob(os.path.join(CASE_PATH, "PTV*.csv"))
    main_ptv_path = sorted(ptv_files)[-1] 
    print(f"Main PTV for Normalization: {os.path.basename(main_ptv_path)}")
    
    main_ptv_mask = load_sparse_csv(main_ptv_path, is_structure=True)
    if dose_grid is None or main_ptv_mask is None: return

    # 归一化逻辑
    ptv_indices = main_ptv_mask > 0.5
    if not np.any(ptv_indices): return
    max_ptv_dose = np.max(dose_grid[ptv_indices])
    scaling_factor = 1.08 / max_ptv_dose
    normalized_dose = dose_grid * scaling_factor
    print(f"Dose Normalized. Factor: {scaling_factor:.4f}")

    # ---------------------------------------------------------
    # 2. 构建 Objective Map
    # ---------------------------------------------------------
    objective_map = np.zeros(FULL_SHAPE, dtype=np.float32)
    
    # 获取目录下所有 csv
    all_files = glob.glob(os.path.join(CASE_PATH, "*.csv"))
    
    # 按照优先级排序：我们希望 PTV 最后写入（覆盖 OAR），或者 OAR 覆盖 PTV？
    # 通常 PTV 是最高优先级目标。
    # 我们先写 OAR，再写 PTV，这样 PTV 会覆盖重叠区域。
    sorted_files = sorted(all_files, key=lambda x: "PTV" in os.path.basename(x))
    
    loaded_structures = []
    
    for fpath in sorted_files:
        fname = os.path.basename(fpath).replace(".csv", "")
        
        # 检查这个文件是否在我们定义的映射表中
        # 比如 fname="Brainstem" -> 在 Mapping 中
        target_val = None
        for key, val in OBJECTIVE_MAPPING.items():
            if key in fname:
                target_val = val
                break
        
        if target_val is not None:
            print(f"Adding structure {fname} with Objective Value {target_val}")
            mask = load_sparse_csv(fpath, is_structure=True)
            if mask is not None:
                # 将该结构区域赋值为 target_val
                # 注意：这里直接覆盖 (Override)
                objective_map[mask > 0.5] = target_val
                loaded_structures.append(fname)
        else:
            # print(f"Skipping {fname} (not in objective list)")
            pass

    # ---------------------------------------------------------
    # 3. 计算 State Frame 1
    # ---------------------------------------------------------
    # Definition: Normalized Dose - Objective Map
    # Filter: "setting the voxels without an associated dose objective value to zero"
    
    # 也就是：只在 objective_map > 0 的地方计算差值，其他地方为 0
    roi_mask = objective_map > 0
    
    state_frame_1_raw = np.zeros_like(normalized_dose)
    state_frame_1_raw[roi_mask] = normalized_dose[roi_mask] - objective_map[roi_mask]
    
    print("State Frame 1 (Raw Difference) Computed.")
    print(f"Min Diff: {np.min(state_frame_1_raw)}, Max Diff: {np.max(state_frame_1_raw)}")

    # ---------------------------------------------------------
    # 4. 论文特定的预处理 (可选，为了完全复现网络输入)
    # ---------------------------------------------------------
    # 论文[cite: 121]: "For the first array, a constant 1 was added... multiplied by 50."
    # 这样是为了把浮点数转成 0-255 的 uint8 图片
    state_frame_1_uint8 = (state_frame_1_raw + 1.0) * 50
    state_frame_1_uint8 = np.clip(state_frame_1_uint8, 0, 255)
    
    # ---------------------------------------------------------
    # 5. 可视化
    # ---------------------------------------------------------
    # 选取包含 PTV 重心的切片
    coords = np.argwhere(main_ptv_mask > 0)
    z_slice = int(np.mean(coords[:, 0])) if len(coords) > 0 else 64

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 左上: Normalized Dose
    axes[0,0].set_title(f"1. Normalized Dose (Slice {z_slice})")
    im1 = axes[0,0].imshow(normalized_dose[z_slice, :, :], cmap='jet', vmin=0, vmax=1.2)
    plt.colorbar(im1, ax=axes[0,0])
    
    # 右上: Objective Map (这是我们要验证的新东西)
    axes[0,1].set_title("2. Objective Map (Targets)")
    # 使用特殊的 cmap 区分不同数值
    im2 = axes[0,1].imshow(objective_map[z_slice, :, :], cmap='tab20c', vmin=0, vmax=1.2)
    plt.colorbar(im2, ax=axes[0,1])

    # 左下: State Frame 1 (Raw Difference)
    axes[1,0].set_title("3. State Frame 1 (Dose - Objective)")
    # 差值可能是负的（剂量不足）也可能是正的（剂量超标）
    # 使用 coolwarm 色图，0 是白色
    im3 = axes[1,0].imshow(state_frame_1_raw[z_slice, :, :], cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im3, ax=axes[1,0])

    # 右下: State Frame 1 (Preprocessed uint8)
    axes[1,1].set_title("4. Network Input (uint8, scaled)")
    im4 = axes[1,1].imshow(state_frame_1_uint8[z_slice, :, :], cmap='gray', vmin=0, vmax=255)
    plt.colorbar(im4, ax=axes[1,1])

    save_path = os.path.join(OUTPUT_DIR, "step2_state1_visualization.png")
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    main()