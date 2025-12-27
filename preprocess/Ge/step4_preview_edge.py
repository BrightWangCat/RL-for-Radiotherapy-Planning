import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import binary_dilation

# =================配置区域=================
CASE_PATH = "/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans/pt_241"
OUTPUT_DIR = "./output_step4_edge_preview"
FULL_SHAPE = (128, 128, 128)
TOTAL_VOXELS = 128 * 128 * 128

# 重新定义映射表 (保持一致性)
OBJECTIVE_MAPPING = {
    "PTV70": 1.0, "PTV63": 0.9, "PTV56": 0.8,
    "Brainstem": 0.7, "SpinalCord": 0.64,
    "RightParotid": 0.37, "LeftParotid": 0.37,
    "Esophagus": 0.6, "Larynx": 0.6, "Mandible": 1.0,
}
# =========================================

def load_sparse_csv(csv_path, is_structure=False):
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
    except: return None

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"Processing case: {CASE_PATH}")

    # 1. Load Dose & PTV
    dose_path = os.path.join(CASE_PATH, "dose.csv")
    dose_grid = load_sparse_csv(dose_path)
    
    ptv_files = glob.glob(os.path.join(CASE_PATH, "PTV*.csv"))
    main_ptv_path = sorted(ptv_files)[-1] 
    main_ptv_mask = load_sparse_csv(main_ptv_path, is_structure=True)
    
    if dose_grid is None or main_ptv_mask is None: return

    # 2. Normalize
    ptv_indices = main_ptv_mask > 0.5
    max_ptv_dose = np.max(dose_grid[ptv_indices])
    scaling_factor = 1.08 / max_ptv_dose
    normalized_dose = dose_grid * scaling_factor
    print(f"Normalization complete. Max PTV Dose: {np.max(normalized_dose[ptv_indices]):.4f}")

    # 3. Build Objective Map
    objective_map = np.zeros(FULL_SHAPE, dtype=np.float32)
    sorted_files = sorted(glob.glob(os.path.join(CASE_PATH, "*.csv")), key=lambda x: "PTV" in os.path.basename(x))
    
    for fpath in sorted_files:
        fname = os.path.basename(fpath).replace(".csv", "")
        target_val = None
        for key, val in OBJECTIVE_MAPPING.items():
            if key in fname: target_val = val; break
        
        if target_val is not None:
            mask = load_sparse_csv(fpath, is_structure=True)
            if mask is not None: objective_map[mask > 0.5] = target_val

    # 4. State Frame 1 Calculation (With Edge Enhancement)
    # 取一个切片演示 (z=53 from previous run)
    coords = np.argwhere(main_ptv_mask > 0)
    z_slice = int(np.mean(coords[:, 0])) if len(coords) > 0 else 64
    
    # 切片级操作 (模拟网络输入生成过程)
    dose_slice = normalized_dose[z_slice, :, :]
    obj_slice = objective_map[z_slice, :, :]
    ptv_slice_mask = main_ptv_mask[z_slice, :, :] > 0.5

    # Step A: Difference
    # 只在有目标定义的区域计算差值
    roi_mask = obj_slice > 0
    diff_map = np.zeros_like(dose_slice)
    diff_map[roi_mask] = dose_slice[roi_mask] - obj_slice[roi_mask]

    # Step B: Scaling (float -> uint8 space)
    # "constant 1 was added ... multiplied by 50"
    # Result range: -1.0 diff -> 0, +1.0 diff -> 100
    state_frame_scaled = (diff_map + 1.0) * 50
    state_frame_scaled = np.clip(state_frame_scaled, 0, 255) # 基础层

    # Step C: Edge Enhancement (The New Part!)
    # "All voxels surrounding the PTV were set to 255"
    # 使用形态学膨胀 (Dilation) 找外轮廓
    # boundary = dilated_mask - original_mask
    dilated_mask = binary_dilation(ptv_slice_mask, iterations=1)
    boundary_mask = dilated_mask & (~ptv_slice_mask)
    
    # 将轮廓位置设为 255
    final_state_frame = state_frame_scaled.copy()
    final_state_frame[boundary_mask] = 255.0

    # 5. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Raw Difference
    im1 = axes[0].imshow(diff_map, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title("1. Raw Difference (Dose - Obj)")
    plt.colorbar(im1, ax=axes[0])
    
    # 2. Scaled (Before Edge)
    im2 = axes[1].imshow(state_frame_scaled, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("2. Scaled (uint8 space)")
    plt.colorbar(im2, ax=axes[1])
    
    # 3. Final with Edge (PTV Boundary=255)
    im3 = axes[2].imshow(final_state_frame, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title("3. Final Input (With 255 Boundary)")
    plt.colorbar(im3, ax=axes[2])
    
    # 画个箭头指出高亮处
    # 简单找个边界点
    ys, xs = np.where(boundary_mask)
    if len(xs) > 0:
        axes[2].annotate('PTV Boundary (255)', xy=(xs[0], ys[0]), xytext=(xs[0]+20, ys[0]-20),
                         arrowprops=dict(facecolor='red', shrink=0.05), color='red')

    save_path = os.path.join(OUTPUT_DIR, "step4_edge_check.png")
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    main()